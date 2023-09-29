

from models import create_model
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
from prettytable import PrettyTable


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



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

args = parse_args()
cfg = Config.fromfile(args.config)
add_args_to_cfg(args, cfg)
model = create_model(cfg.model_type, cfg.model_cfg)

count_parameters(model)