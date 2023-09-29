from torch.utils.data import DataLoader

from .oxford import OxfordDataset
from .umd import UMDDataset
from .residential import ResidentialDataset
from .business import BusinessDataset
from .university import UniversityDataset
from .utils.collate_fns import create_collate_fn
from .utils.batch_samplers import ExpansionBatchSampler


def create_dataloaders(dataset_type, cfg, subset_types, log, debug=False):

    type2dataset = dict(
        Oxford=OxfordDataset,
        Umd=UMDDataset,
        Residential=ResidentialDataset,
        Business=BusinessDataset,
        University=UniversityDataset,
    )
    dataset = type2dataset[dataset_type](cfg.dataset_cfg, log, debug)
    dataloaders = []
    s_types = [subset_types] if type(subset_types) == str else subset_types

    for subset_type in s_types:
        subset = dataset.subset(subset_type)
        if subset_type == 'train':
            type2sampler = dict(
                ExpansionBatchSampler=ExpansionBatchSampler
            )
            batch_sampler = type2sampler[cfg.train_cfg.batch_sampler_type](
                dataset=subset, cfg=cfg.train_cfg.batch_sampler_cfg, log=log
            )
            collate_fn = create_collate_fn(
                subset, cfg.model_cfg.quantization_size if hasattr(cfg.model_cfg, 'quantization_size') else None, cfg.model_cfg.ndt if hasattr(
                cfg.model_cfg, 'ndt') else None, True)
            dataloader = DataLoader(
                subset,
                batch_sampler=batch_sampler,
                pin_memory=True,
                collate_fn=collate_fn,
                num_workers=cfg.train_cfg.num_workers
            )
            dataloaders.append((dataloader, batch_sampler))
        else:
            collate_fn = create_collate_fn(subset, cfg.model_cfg.quantization_size if hasattr(
                cfg.model_cfg, 'quantization_size') else None, cfg.model_cfg.ndt if hasattr(
                cfg.model_cfg, 'ndt') else None, False)
            dataloader = DataLoader(
                subset,
                batch_size=cfg.eval_cfg.batch_sampler_cfg.batch_size,
                shuffle=False,
                pin_memory=False,
                collate_fn=collate_fn,
                num_workers=cfg.eval_cfg.num_workers
            )
            dataloaders.append((dataloader, None))

    return dataloaders[0] if type(subset_types) == str else dataloaders
