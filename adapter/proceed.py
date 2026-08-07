
import torch
import torch.nn as nn
import transformers
from adapter.module.generator import AdaptGenerator
from adapter.module import down_up


def normalize(W, max_norm=1):
    W_norm = torch.norm(W, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / W_norm, max=1)
    return W * scale


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Proceed(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        if args.freeze:
            backbone.requires_grad_(False)
        self.backbone = add_adapters_(backbone, args)
        # Let the backbone re-enforce its own internal freeze policy after adapters
        # are injected (e.g. TTM always keeps its pretrained backbone frozen).
        if hasattr(self.backbone, 'post_proceed_init'):
            self.backbone.post_proceed_init()
        self.more_bias = not args.freeze
        # --- concept-encoder ablation mode (see documents/THESIS_REPORT_NOTES.md) ---
        #   dual    : two encoders, drift = c(X_t) - c(X_{t-H})   [default / original]
        #   shared  : one shared encoder for both (recent truncated to seq_len)
        #   current : only c(X_t), NO drift subtraction
        #   stats   : data-space per-channel moment featurizer, drift on stats
        self.concept_mode = getattr(args, 'concept_mode', 'dual')
        self.n_stats = 4  # mean, std, last, trend (data-space featurizer)
        concept_features = args.enc_in * self.n_stats if self.concept_mode == 'stats' else args.concept_dim
        self.generator = AdaptGenerator(backbone, concept_features,
                                        activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
                                        adaptive_dim=False, need_bias=self.more_bias,
                                        shared=not args.individual_generator,
                                        mid_dim=args.bottleneck_dim)
        # print(self.adapters)
        self.register_buffer('recent_batch', torch.zeros(1, args.seq_len + args.pred_len, args.enc_in), persistent=False)
        if args.ema > 0:
            self.register_buffer('recent_concept', None, persistent=True)
        if self.concept_mode in ('dual', 'shared', 'current'):
            self.mlp1 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len, args.concept_dim), nn.GELU(),
                                      nn.Linear(args.concept_dim, args.concept_dim))
        if self.concept_mode == 'dual':
            self.mlp2 = nn.Sequential(Transpose(-1, -2), nn.Linear(args.seq_len + args.pred_len, args.concept_dim), nn.GELU(),
                                      nn.Linear(args.concept_dim, args.concept_dim))
        self.ema = args.ema
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

    def _stats(self, x):
        # Data-space per-channel featurizer: [mean, std, last, trend] -> (..., n_stats*C)
        mean = x.mean(-2)
        std = x.std(-2)
        last = x[..., -1, :]
        L = x.shape[-2]
        t = torch.linspace(-1.0, 1.0, L, device=x.device, dtype=x.dtype).reshape(*([1] * (x.dim() - 2)), L, 1)
        tc = t - t.mean(-2, keepdim=True)
        xc = x - x.mean(-2, keepdim=True)
        slope = (tc * xc).mean(-2) / (tc.pow(2).mean(-2) + 1e-8)
        return torch.cat([mean, std, last, slope], dim=-1)

    def generate_adaptation(self, x):
        if self.concept_mode == 'current':
            # only the current concept c_t, NO drift subtraction (ablation)
            concept = self.mlp1(x).mean(-2)
            return self.generator(concept, need_clip=not self.args.wo_clip)

        if self.concept_mode == 'stats':
            concept = self._stats(x)
            recent_concept = self._stats(self.recent_batch)
            recent_concept = recent_concept.mean(list(range(0, recent_concept.dim() - 1)))
        elif self.concept_mode == 'shared':
            concept = self.mlp1(x).mean(-2)
            rec_in = self.recent_batch[..., -self.args.seq_len:, :]
            recent_concept = self.mlp1(rec_in).mean(-2).mean(list(range(0, rec_in.dim() - 2)))
        else:  # 'dual' (default / original)
            concept = self.mlp1(x).mean(-2)
            recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(list(range(0, self.recent_batch.dim() - 2)))

        if self.ema > 0:
            if self.recent_concept is not None:
                recent_concept = self.recent_concept * self.ema + recent_concept * (1 - self.ema)
            if self.flag_update or self.flag_online_learning and not self.flag_current:
                self.recent_concept = recent_concept.detach()
        drift = concept - recent_concept
        res = self.generator(drift, need_clip=not self.args.wo_clip)
        return res

    def forward(self, *x):
        if self.flag_basic:
            adaptations = {}
            for i, (k, adapter) in enumerate(self.generator.bottlenecks.items()):
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.generator.dim_name_dict[k])
        else:
            adaptations = self.generate_adaptation(x[0])
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.generator.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])
        if self.args.do_predict:
            return self.backbone(*x)
        else:
            return self.backbone(*x)

    def freeze_adapter(self, freeze=True):
        for module_name in ['mlp1', 'mlp2']:
            if hasattr(self, module_name):
                getattr(self, module_name).requires_grad_(not freeze)
                getattr(self, module_name).zero_grad(set_to_none=True)
        for adapter in self.generator.bottlenecks.values():
            adapter.weights.requires_grad_(not freeze)
            adapter.weights.zero_grad(set_to_none=True)
            adapter.biases[:len(adapter.weights) - 1].requires_grad_(not freeze)
            adapter.biases[:len(adapter.weights) - 1].zero_grad(set_to_none=True)

    def freeze_bias(self, freeze=True):
        if self.more_bias:
            for adapter in self.generator.bottlenecks.values():
                adapter.biases[-1].requires_grad_(not freeze)
                adapter.biases[-1:].zero_grad(set_to_none=True)


def add_adapters_(parent_module: nn.Module, args, top_level=True):
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D,
                                                                     nn.LayerNorm, nn.BatchNorm1d)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            down_up.add_down_up_(parent_module, name, freeze_weight=args.freeze, merge_weights=args.merge_weights,)
        else:
            add_adapters_(module, args, False)
    return parent_module
