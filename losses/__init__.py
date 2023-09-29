from .contrastive_loss import BatchHardContrastiveLossWithMasks
from .triplet_loss import BatchHardTripletLossWithMasks


def create_loss_fn(loss_type, cfg):

    type2loss_fn = dict(
        BatchHardTripletMarginLoss=BatchHardTripletLossWithMasks,
        BatchHardContrastiveLoss=BatchHardContrastiveLossWithMasks
    )

    return type2loss_fn[loss_type](cfg)
