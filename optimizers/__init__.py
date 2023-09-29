import torch


def Adam(cfg, params):
    return torch.optim.Adam(
        params=params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas
    )


def create_optimizer(optimizer_type, cfg, params):
    type2optimizer = dict(
        Adam=Adam
    )
    return type2optimizer[optimizer_type](cfg, params)
