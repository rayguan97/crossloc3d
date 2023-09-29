_base_ = [
    './base_cfg.py',
    './dataset_cfgs/oxford_cfg.py'
]

task_type = 'ours_me'

optimizer_type = 'Adam'
optimizer_cfg = dict(
    lr=2e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
)

scheduler_type = 'MultiStepLR'
scheduler_cfg = dict(
    gamma=0.1,
    milestones=(80, 120, 160)
)

end_epoch = 200

train_cfg = dict(
    save_per_epoch=10,
    val_per_epoch=5,
    batch_sampler_type='ExpansionBatchSampler',
    batch_sampler_cfg=dict(
        max_batch_size=128,
        batch_size_expansion_rate=1.4,
        batch_expansion_threshold=0.7,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    ),
    num_workers=0,
)

eval_cfg = dict(
    batch_sampler_cfg=dict(
        batch_size=32,
        drop_last=False,
    ),
    num_workers=0,
    normalize_embeddings=False,
)

model_type = 'Ours'
model_cfg = dict(
    backbone_cfg=dict(
        up_conv_cfgs=[
            [dict(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
            ),
            dict(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
            )],
            [dict(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
            ),
            dict(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
            )],
            [dict(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
            ),
            dict(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
            )],
        ],
        transformer_cfg=dict(
            num_attn_layers=6,
            global_channels=64,
            local_channels=0,
            num_centers=[256, 128, 128, 64, 32, 32],
            num_heads=4,
            time_dim=8,
            learned_sinusoidal_cond=True,
        ),
        pointnet_cfg=dict(
            std_cfg=dict(
                conv_channels=[64, 128, 512],
                fc_channels=[256, 128]
            ),
            global_feat = True,
            channels=[64, 128, 256]
        ),
        in_channels=1,
        out_channels=512,
        fine_to_coarse = False,
        step_size=2,
    ),
    pool_cfg=dict(
        type='NetVlad',
        in_channels=512,
        out_channels=512,
        cluster_size=64,
        gating=True,
        add_bn=True
    ),
    quantization_size=[0.01, 0.12, 0.2],
)
