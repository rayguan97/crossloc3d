from .ours.model import Ours



def create_model(model_type, model_cfg):
    type2model = dict(
        Ours=Ours,
    )
    return type2model[model_type](model_cfg)
