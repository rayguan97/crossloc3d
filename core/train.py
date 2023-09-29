from tqdm import tqdm
import torch
import os.path as osp
from time import time

from datasets import create_dataloaders
from tasks import create_task
from optimizers import create_optimizer
from schedulers import create_scheduler
from utils import AverageValue, Metrics
from .val import val


def train(cfg, log):

    train_data_loader, train_batch_sampler = create_dataloaders(
        dataset_type=cfg.dataset_type,
        cfg=cfg,
        subset_types='train',
        log=log,
        debug=cfg.debug
    )

    task = create_task(cfg.task_type, cfg, log)

    if torch.cuda.is_available():
        task.cuda()

    if cfg.resume_from is not None:
        task_state = task.load(cfg.resume_from)
        start_epoch = task_state['epoch'] if 'epoch' in cfg.resume_items else 0
        optimizer = create_optimizer(
            cfg.optimizer_type, cfg.optimizer_cfg, task.model_params())
        scheduler = create_scheduler(
            cfg.scheduler_type, cfg.scheduler_cfg, optimizer)
        if 'optim' in cfg.resume_items:
            optimizer.load_state_dict(task_state['optim_state_dict'])
        if 'sched' in cfg.resume_items:
            scheduler.load_state_dict(task_state['sched_state_dict'])
        if 'sampler' in cfg.resume_items and hasattr(train_batch_sampler, 'load_state_dict'):
            train_batch_sampler.load_state_dict(
                task_state['sampler_state_dict'])
        best_metrics = Metrics(
            'Recall@1%', task_state['best_metrics']) if 'metrics' in cfg.resume_items else None
    else:
        start_epoch = 0
        optimizer = create_optimizer(
            cfg.optimizer_type, cfg.optimizer_cfg, task.model_params())
        scheduler = create_scheduler(
            cfg.scheduler_type, cfg.scheduler_cfg, optimizer)
        best_metrics = None

    end_epoch = cfg.end_epoch

    for epoch in range(start_epoch + 1, end_epoch + 1):

        log.info('[Epoch%4d/%4d] Start training ...' % (epoch, end_epoch))
        epoch_start_time = time()
        losses = None

        task.before_epoch(epoch)
        task.train()

        with tqdm(total=len(train_data_loader)) as pbar:
            for batch_idx, (meta, data) in enumerate(train_data_loader):

                optimizer.zero_grad()
                loss_info = task.step(meta, data)
                if losses is None:
                    losses = AverageValue(list(loss_info.keys()))
                losses.update(loss_info)
                optimizer.step()
                torch.cuda.empty_cache()
                details = {}
                details.update(losses.avg() if type(losses.avg())
                               == dict else {'loss': losses.avg()})
                pbar.set_postfix(**details)
                pbar.update(1)

                if cfg.debug:
                    break
        scheduler.step()

        epoch_end_time = time()

        log.info(
            '[Epoch%4d/%4d] Training time= %.3fs %s' %
            (epoch, end_epoch, epoch_end_time - epoch_start_time, losses.avg_str()))

        save_dir = osp.join(cfg.work_dir, 'ckpt_tmp.pth')
        task.save(
            save_dir=save_dir,
            epoch=epoch,
            best_metrics=None,
            optim_state_dict=optimizer.state_dict(),
            sched_state_dict=scheduler.state_dict(),
            sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                train_batch_sampler, 'state_dict') else None
        )

        if epoch % cfg.train_cfg.val_per_epoch == 0 or epoch == end_epoch:
            log.info('[Epoch%4d/%4d] Start validating ...' %
                     (epoch, end_epoch))
            epoch_start_time = time()
            metrics = val(cfg, log, task)
            epoch_end_time = time()
            log.info(
                '[Epoch%4d/%4d] Validating time= %.3fs' %
                (epoch, end_epoch, epoch_end_time - epoch_start_time))

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                save_dir = osp.join(cfg.work_dir, 'best_ckpt.pth')
                task.save(
                    save_dir=save_dir,
                    epoch=epoch,
                    best_metrics=metrics.state_dict(),
                    optim_state_dict=optimizer.state_dict(),
                    sched_state_dict=scheduler.state_dict(),
                    sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                        train_batch_sampler, 'state_dict') else None
                )

            if epoch % cfg.train_cfg.save_per_epoch == 0 or epoch == end_epoch:
                save_dir = osp.join(cfg.work_dir, 'ckpt[epoch=%d].pth' % epoch)
                task.save(
                    save_dir=save_dir,
                    epoch=epoch,
                    best_metrics=metrics.state_dict(),
                    optim_state_dict=optimizer.state_dict(),
                    sched_state_dict=scheduler.state_dict(),
                    sampler_state_dict=train_batch_sampler.state_dict() if hasattr(
                        train_batch_sampler, 'state_dict') else None
                )
        if train_batch_sampler is not None:
            train_batch_sampler.update(losses.avg())

        task.after_epoch(epoch)

        if cfg.debug:
            break
