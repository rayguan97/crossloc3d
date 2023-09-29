from .default_me_task import DefaultMETask
from .ours_me_task import OursMETask


def create_task(task_type, cfg, log):
    type2task = dict(
        default_me=DefaultMETask,
        ours_me=OursMETask,
    )
    return type2task[task_type](cfg, log)
