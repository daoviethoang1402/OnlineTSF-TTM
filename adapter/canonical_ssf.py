import torch.nn as nn
import transformers

from adapter.module import ssf


def add_ssf_adapters_(parent_module: nn.Module, args, top_level=True):
    """Injects canonical *learnable* SSF (adapter.module.ssf.SSF: out-scale + shift,
    no input scale) in place of PROCEED's drift-conditioned Down_Up.

    Mirrors adapter.proceed.add_adapters_'s module-selection logic exactly (same
    --tune_mode semantics: 'down_up' wraps Linear/Conv1d/Conv1D only, 'all_down_up'
    also wraps LayerNorm/BatchNorm1d) so the set of adapted layers -- and therefore
    the comparison to a PROCEED run at the same --tune_mode -- is apples-to-apples.

    freeze_weight is always True here (not tied to args.freeze): this module is a
    frozen-backbone-only baseline by construction, matching the reduced-noft /
    frozen-PROCEED family it is meant to be compared against.
    """
    for name, module in parent_module.named_children():
        if args.tune_mode == 'all_down_up' and isinstance(
                module, (nn.Conv1d, nn.Linear, transformers.Conv1D, nn.LayerNorm, nn.BatchNorm1d)):
            ssf.add_ssf_(parent_module, name, freeze_weight=True, merge_weights=args.merge_weights,
                        learnable=True)
        elif args.tune_mode == 'down_up' and isinstance(module, (nn.Conv1d, nn.Linear, transformers.Conv1D)):
            ssf.add_ssf_(parent_module, name, freeze_weight=True, merge_weights=args.merge_weights,
                        learnable=True)
        else:
            add_ssf_adapters_(module, args, False)
    return parent_module


class CanonicalSSF(nn.Module):
    """
    Canonical learnable SSF (Lian et al., "Scaling & Shifting Your Features",
    NeurIPS 2022): raw per-layer output-scale + shift, identity-initialized,
    trained end-to-end by backprop -- no hypernetwork, no concept encoders,
    no drift signal of any kind.

    This is the "is it just plain SSF?" control for the PROCEED ablation: same
    wrapped-layer set and the same frozen-backbone policy as
    Proceed(tune_mode=..., static_ssf=True, ssf_no_input_scale=True), but here
    scale/shift are trained directly by the online optimizer rather than being
    generated from a (possibly zeroed) drift signal by a hypernetwork. Compare
    against that static_ssf run to separate two different questions:
      - static_ssf=True: what does the existing hypernetwork settle on when the
        drift input is zeroed, with all its machinery (encoders, W_down, bottleneck)
        still attached but functionally unused?
      - CanonicalSSF: what is the minimal object -- raw per-layer scale/shift and
        nothing else -- capable of, with no dead machinery attached at all?

    Backbone is frozen unconditionally (not gated on args.freeze); the online
    optimizer only ever sees the injected scale/shift parameters.
    """

    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        backbone.requires_grad_(False)
        self.backbone = add_ssf_adapters_(backbone, args)
        # Some backbones (e.g. TinyTimeMixer) override requires_grad_() with extra
        # invariants (backbone core always frozen regardless of caller); re-run it
        # after adapter injection in case add_ssf_adapters_' new modules disturbed
        # anything, mirroring Proceed's post_proceed_init() call.
        if hasattr(self.backbone, 'post_proceed_init'):
            self.backbone.post_proceed_init()

    def forward(self, *x):
        return self.backbone(*x)
