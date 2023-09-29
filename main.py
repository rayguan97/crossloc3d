import argparse
import os
import os.path as osp
from datetime import datetime
import torch
import numpy as np
import random
import torch.multiprocessing

from utils import Config, Logger
from core import train, val


def parse_args():
    parser = argparse.ArgumentParser(
        description='point cloud retrieval task')
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--workspace', type=str,
                        default='./workspace', help='path to workspace')
    parser.add_argument('--resume_from', type=str,
                        default=None, help='path to checkpoint file')
    parser.add_argument('--resume_items', nargs='+', type=str, default=[
                        'model', 'epoch', 'metrics', 'optim', 'sched', 'sampler', 'all', 'no_metrics'], help='choose which component of checkpoint to resume, including model, epoch, optim, sched, sampler, metrics, or all')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='choose whether to set deterministic options for CUDNN backend')
    parser.add_argument('--mode', type=str, default='train', choices=[
                        'train', 'val'], help='choose which mode to run point cloud retrieval')
    parser.add_argument('--debug', action='store_true',
                        help='choose which state to run point cloud retrieval')
    args = parser.parse_args()
    return args


def add_args_to_cfg(args, cfg):
    cfg.work_dir = osp.abspath(osp.join(args.workspace, osp.splitext(osp.basename(args.config))[
        0], datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')))
    cfg.resume_from = osp.abspath(
        args.resume_from) if args.resume_from is not None else None
    cfg.resume_items = args.resume_items
    if cfg.resume_items == ['all']:
        cfg.resume_items = ['model', 'epoch',
                            'metrics', 'optim', 'sched', 'sampler']
    elif cfg.resume_items == ['no_metrics']:
        cfg.resume_items = ['model', 'epoch', 'optim', 'sched', 'sampler']
    cfg.mode = args.mode
    cfg.debug = args.debug
    if cfg.debug:
        cfg.seed = 1234
        cfg.deterministic = True
        if hasattr(cfg, 'train_cfg'):
            cfg.train_cfg.val_per_epoch = 1
            if hasattr(cfg.train_cfg, 'batch_sampler_cfg'):
                cfg.train_cfg.batch_sampler_cfg.batch_size = cfg.train_cfg.batch_sampler_cfg.max_batch_size

    else:
        cfg.seed = args.seed
        cfg.deterministic = args.deterministic


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    add_args_to_cfg(args, cfg)

    torch.multiprocessing.set_sharing_strategy('file_system')

    if not cfg.debug:
        os.makedirs(cfg.work_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if not cfg.debug:
        cfg.dump(osp.join(cfg.work_dir, 'config.py'))

    # init log
    log_file_dir = osp.join(
        cfg.work_dir, 'Retrieval3D.log') if not cfg.debug else None
    log = Logger(name='R3D', log_file=log_file_dir)
    log.info(f'Config : \n{cfg.pretty_text}')

    if cfg.mode == 'train':
        train(cfg, log)
    elif cfg.mode == 'val':
        val(cfg, log)
    else:
        raise NotImplementedError(
            '{} has not been implemented!'.format(cfg.mode))


if __name__ == '__main__':
    main()
