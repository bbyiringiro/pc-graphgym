import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_network
import predictive_coding as pc




from dataclasses import dataclass, replace
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    # GeneralLayer,
    # LayerConfig,
    GeneralMultiLayer,
    # new_layer_config,
)
from torch_geometric.graphgym.register import register_stage


@dataclass
class LayerConfig:
    # batchnorm parameters.
    has_batchnorm: bool = False
    bn_eps: float = 1e-5
    bn_mom: float = 0.1

    # mem parameters.
    mem_inplace: bool = False

    # gnn parameters.
    dim_in: int = -1
    dim_out: int = -1
    edge_dim: int = -1
    dim_inner: int = None
    num_layers: int = 2
    has_bias: bool = True
    # regularizer parameters.
    has_l2norm: bool = True
    dropout: float = 0.0
    # activation parameters.
    has_act: bool = True
    final_act: bool = True
    act: str = 'relu'

    # other parameters.
    keep_edge: float = 0.5
    inter_layer_pc:bool = False


def new_layer_config(dim_in, dim_out, num_layers, has_act, has_bias, cfg):
    return LayerConfig(
    has_batchnorm=cfg.gnn.batchnorm,
    bn_eps=cfg.bn.eps,
    bn_mom=cfg.bn.mom,
    mem_inplace=cfg.mem.inplace,
    dim_in=dim_in,
    dim_out=dim_out,
    edge_dim=cfg.dataset.edge_dim,
    has_l2norm=cfg.gnn.l2norm,
    dropout=cfg.gnn.dropout,
    has_act=has_act,
    final_act=True,
    act=cfg.gnn.act,
    has_bias=has_bias,
    keep_edge=cfg.gnn.keep_edge,
    dim_inner=cfg.gnn.dim_inner,
    num_layers=num_layers,
    inter_layer_pc=cfg.pc.use_pc,
)



class GeneralLayer2(nn.Module):
    """
    General wrapper for layers
    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation after the layer
        has_bn (bool):  Whether has BatchNorm in the layer
        has_l2norm (bool): Wheter has L2 normalization after the layer
        **kwargs (optional): Additional args
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps,
                               momentum=layer_config.bn_mom))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=layer_config.dropout,
                           inplace=layer_config.mem_inplace))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act])
        if layer_config.inter_layer_pc:
                layer_wrapper.append(pc.PCLayer())
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch



def GNNLayer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    return GeneralLayer2(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg))


def GNNPreMP(dim_in, dim_out, num_layers):
    """
    Wrapper for NN layer before GNN message passing

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer(
        'linear',
        layer_config=new_layer_config(dim_in, dim_out, num_layers,
                                      has_act=False, has_bias=False, cfg=cfg))


# class MyPCLayer(pc.PCLayer):
#     def __init__(self):
#         super(MyPCLayer, self).__init__()
#         self.pc_layer=pc.PCLayer()
#     def forward(self, batch):
#         batch.x = self.pc_layer(batch.x)
#         return batch


@register_stage('my_stack')
@register_stage('my_skipsum')
@register_stage('my_skipconcat')
class GNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'my_skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'my_skipsum':
                if batch.x.shape[-1]==x.shape[-1]: # TASK makes sure the layer mp layer have same dims .. TASK
                    batch.x = x + batch.x
            elif cfg.gnn.stage_type == 'my_skipconcat' and \
                    i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch


class FeatureEncoder(nn.Module):
    """
    Encoding node and edge features
    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        """"""
        for module in self.children():
            batch = module(batch)
        return batch

@register_network('pc_network')
class GNN(nn.Module):
    """
    General GNN model: encoder + stage + head
    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        """"""
        for module in self.children():
            batch = module(batch)
        return batch
