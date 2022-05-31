import torch.nn as nn
from torch.nn import functional as F 



import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head



@register_head('myhead')
class CustomNodeHead(nn.Module):
    """
    GNN prediction head for node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, out, batch):
        mask = '{}_mask'.format(batch.split)
        return F.softmax(out[batch[mask]], dim=1), \
            batch.y[batch[mask]]

    def forward(self, x, batch):
        out = self.layer_post_mp(x)
        pred, label = self._apply_index(out, batch)
        return pred, label