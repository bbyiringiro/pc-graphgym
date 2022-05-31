import torch.optim as optim
from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


# @register_config('my_config')
@register_config('my_config')


def set_cfg(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # # example argument
    # cfg.use_pc = 'example'

    # example argument group
    # cfg.example_group = CN()

    # then argument can be specified within the group
    # cfg.example_group.example_arg = 'example'

    #predictive coding
    cfg.pc = CN()
    cfg.pc.use_pc = False

    cfg.pc.T=12
    cfg.pc.update_x_at='all'
    cfg.pc.optimizer_x_fn='Adam'
    cfg.pc.optimizer_x_lr =0.5
    cfg.pc.update_p_at='all'
    cfg.pc.optimizer_p_fn='Adam'
    cfg.pc.optimizer_p_lr =0.001
