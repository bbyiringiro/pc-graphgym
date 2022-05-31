import logging
import time

import torch

from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
import predictive_coding as pc

from torch_geometric.graphgym.utils.epoch import (
    is_ckpt_epoch,
    is_eval_epoch,
    is_train_eval_epoch,
)

mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)
criterion = torch.nn.CrossEntropyLoss()

def loss_fn(output, pred_score_res, true_res):
    pred, true = output
    pred_score_res.append(pred)
    true_res.append(true)
    # temporary using mse_loss here because high memory requirement for link_pred task
    if cfg.dataset.task =='edge' or cfg.dataset.task =='link_pred': # TASK 
        return mse_loss(pred, true)
    else:
        # return criterion(output[_train_mask], _target[_train_mask])
        # num_out_features =  pred.shape[-1
        true_one_hot = torch.zeros((true.shape[0], pred.shape[-1]),device=cfg.device).scatter_(1, true.reshape(true.shape[0],1), 1)
        # print(true_one_hot.shape)
        
        # target_one_hot = target_one_hot.to(cfg.device)
        # print((output[_train_mask] - target_one_hot[_train_mask]).pow(2).sum() * 0.5)
        return (pred - true_one_hot).pow(2).sum() * 0.5

def callback_after_t(pc_trainer, batch, orginal_x):
    batch.x = orginal_x # reset the input
def train_epoch(logger, loader,  pc_trainer, scheduler):
    pc_trainer._model.train()

    time_start = time.time()

    for batch in loader:
    
        # loss = 0.0
        # energy = 0.0
        # overall = 0.0
        batch.split = 'train'
        batch.to(torch.device(cfg.device))

        orginal_x = batch.x
        pred_score_res =[]
        true_res  = []
        res = pc_trainer.train_on_batch(
            inputs=batch,
            loss_fn=loss_fn,
            loss_fn_kwargs={
                'pred_score_res': pred_score_res,
                'true_res':true_res
            },
            callback_after_t=callback_after_t,
            callback_after_t_kwargs={
                'batch':batch,
                'orginal_x':orginal_x,

            },
            is_checking_after_callback_after_t=False,

            # is_unwrap_inputs=True,
            is_log_progress=False,
            is_return_results_every_t=False,
        )

        logger.update_stats(true=true_res[0].detach().cpu(),
                    pred=pred_score_res[0].detach().cpu(), loss=res['loss'][-1],
                    lr=pc_trainer._optimizer_p_kwargs['lr'],
                    time_used=time.time() - time_start,
                    params=cfg.params)
        time_start = time.time()
    # loss +=res['loss'][-1]
    # if use_pc:
    #     energy +=res['energy'][-1]
    # overall +=res['overall'][-1]

        
    # results['loss'].append(loss)
    # results['energy'].append(energy)
    # results['overall'].append(overall)
    # results['test_acc'].append(test(pc_model))
    # results['train_acc'].append(train_test(pc_model))
    


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss = loss_fn((pred, true), [],[])
        pred_score=pred
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()

@register_train('pc_train')
def pc_trainer(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """

    pc_trainer = pc.PCTrainer(
        
        ,
        T=cfg.pc.T if cfg.pc.use_pc else 1, #TASK
        update_x_at=cfg.pc.update_x_at,
        optimizer_x_fn=eval(f'torch.optim.{cfg.pc.optimizer_x_fn}'),
        optimizer_x_kwargs={"lr": cfg.pc.optimizer_x_lr },
        update_p_at=cfg.pc.update_p_at,
        optimizer_p_fn=eval(f'torch.optim.{cfg.pc.optimizer_p_fn}'),
        optimizer_p_kwargs={'lr' : cfg.pc.optimizer_p_lr},
        plot_progress_at=[],
    )


    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('PC - Checkpoint found, Task already done')
    else:
        logging.info('PC - Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], pc_trainer, scheduler)
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                model.eval()
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))