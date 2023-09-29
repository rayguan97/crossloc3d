scheduler_type = 'MultiStepLR'
scheduler_cfg = dict(
    gamma=0.5,
    milestones=(50, 100, 150, 200)
)

end_epoch = 250
