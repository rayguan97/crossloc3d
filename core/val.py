import torch

from datasets import create_dataloaders
from tasks import create_task
from .eval import eval


def val(cfg, log, task=None):

    if task is None:
        task = create_task(cfg.task_type, cfg, log)
        if torch.cuda.is_available():
            task.cuda()
        assert cfg.resume_from is not None
        task_state = task.load(cfg.resume_from)

    log.info(
        '++++++++++++++++  Evaluating %s dataset  ++++++++++++++++' % cfg.dataset_type)

    (db_data_loader, _), (q_data_loader, _) = create_dataloaders(
        dataset_type=cfg.dataset_type,
        cfg=cfg,
        subset_types=('database', 'queries'),
        log=log,
        debug=cfg.debug
    )

    metrics = eval(cfg, log, db_data_loader, q_data_loader, task)
    return metrics
