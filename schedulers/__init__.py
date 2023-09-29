import torch


def MultiStepLR(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=cfg.milestones,
        gamma=cfg.gamma
    )


def StepLR(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.step_size,
        gamma=cfg.gamma
    )


def CosineAnnealingLR(cfg, optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.T_max,
        eta_min=cfg.eta_min
    )


def create_scheduler(scheduler_type, cfg, optimizer):
    type2scheduler = dict(
        MultiStepLR=MultiStepLR,
        CosineAnnealingLR=CosineAnnealingLR,
        StepLR=StepLR
    )
    return type2scheduler[scheduler_type](cfg, optimizer)
