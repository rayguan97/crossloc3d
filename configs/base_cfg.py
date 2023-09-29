_base_ = [
    './optimizer_cfgs/adam_cfg.py',
    './scheduler_cfgs/multi_step_lr_cfg.py',
    './loss_cfgs/triplet_loss_cfg.py',
]

train_cfg = dict(
    save_per_epoch=10,
    val_per_epoch=5,
    batch_sampler_type='ExpansionBatchSampler',
    batch_sampler_cfg=dict(
        max_batch_size=64,
        batch_size_expansion_rate=1.4,
        batch_expansion_threshold=0.7,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    ),
    num_workers=16,
)

eval_cfg = dict(
    batch_sampler_cfg=dict(
        batch_size=32,
        drop_last=False,
    ),
    num_workers=16,
)
