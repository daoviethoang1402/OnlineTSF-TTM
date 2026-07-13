import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class ForecastModel(nn.Module):
    def __init__(self, backbone, num_features, seq_len, process_method='RevIN', **kwargs):
        super().__init__()
        self.backbone = backbone
        if process_method.lower() == 'revin':
            self.processor = RevIN(num_features=num_features, **kwargs)
        elif process_method.lower() == 'dishts':
            self.processor = DishTS(num_features=num_features, seq_len=seq_len, **kwargs)
        else:
            raise NotImplementedError

    def post_proceed_init(self):
        """Forward to self.backbone so Proceed's hasattr(...) check reaches it.

        Without this, Proceed sees a ForecastModel (this wrapper) instead of the
        actual backbone and never calls the backbone's own post_proceed_init(),
        silently skipping any freeze re-enforcement the backbone relies on.
        """
        if hasattr(self.backbone, 'post_proceed_init'):
            self.backbone.post_proceed_init()

    def freeze_head(self):
        """Forward to self.backbone; see post_proceed_init() for why this is needed."""
        if hasattr(self.backbone, 'freeze_head'):
            self.backbone.freeze_head()

    def requires_grad_(self, requires_grad: bool = True):
        """Delegate to self.backbone.requires_grad_ instead of the default
        nn.Module.requires_grad_, which would blanket-set every parameter and
        bypass any backbone-specific freeze policy (e.g. TTM's overridden
        requires_grad_ that keeps its backbone permanently frozen).
        """
        self.backbone.requires_grad_(requires_grad)
        self.processor.requires_grad_(requires_grad)
        return self

    def forward(self, x, *args, process=True, **kwargs):
        if process:
            x = self.processor(x, mode='norm')
        outputs = self.backbone(x, *args, **kwargs)
        if not process:
            return outputs
        if isinstance(outputs, tuple):
            pred = self.processor(outputs[0], mode='denorm')
            return [pred] + [o for o in outputs[1:]]
        else:
            return self.processor(outputs, mode='denorm')


class DishTS(RevIN):
    def __init__(self, num_features: int, eps=1e-8, affine=True, seq_len=None, init='standard', **kwargs):
        super().__init__(num_features, eps, affine, **kwargs)
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(num_features, seq_len, 2) / seq_len)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(num_features, seq_len, 2) / seq_len)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len + torch.rand(num_features, seq_len, 2) / seq_len)

    def _get_statistics(self, x):
        x_transpose = x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        # theta = F.gelu(theta)
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sqrt(torch.sum(torch.pow(x - self.phil, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)
        self.xih = torch.sqrt(torch.sum(torch.pow(x - self.phih, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)

    def _normalize(self, x):
        x = (x - self.phil) / self.xil
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.xih
        x = x + self.phih
        return x