from .base_me_task import BaseMETask
from losses import create_loss_fn
from models import create_model
import torch.nn.parallel as parallel


class DefaultMETask(BaseMETask):
    def __init__(self, cfg, log):
        super().__init__(cfg, log)
        self.loss_fn = create_loss_fn(cfg.loss_type, cfg.loss_cfg)

    def _init_model(self):
        model = create_model(self.cfg.model_type, self.cfg.model_cfg)
        return model

    def _train_step(self, meta_data, batch_data, **kwargs):

        if len(self.devices) > 1 and self.master_device is not None:
            model_replicas = parallel.replicate(self.model, self.devices)
            embeddings = parallel.parallel_apply(
                model_replicas, [((pcd, raw_pcd),) for pcd, raw_pcd in zip(batch_data['pcd'], batch_data['raw_pcd'])], devices=self.devices)
            embeddings = parallel.gather(embeddings, self.master_device, dim=0)
        else:
            embeddings = self.model((batch_data['pcd'][0], batch_data['raw_pcd'][0]))
        loss, loss_stats, _ = self.loss_fn(
            embeddings=embeddings,
            positives_mask=batch_data['pos_mask'],
            negatives_mask=batch_data['neg_mask']
        )
        loss.backward()
        return {
            'loss': loss_stats['loss'],
            'mean_pos_pair_dist': loss_stats['mean_pos_pair_dist'],
            'mean_neg_pair_dist': loss_stats['mean_neg_pair_dist'],
            'num_non_zero_triplets': loss_stats['num_non_zero_triplets'],
            'num_triplets': loss_stats['num_triplets']
        }

    def _eval_step(self, meta_data, batch_data, **kwargs):
        if len(self.devices) > 1 and self.master_device is not None:
            model_replicas = parallel.replicate(
                self.model, [self.master_device])
            embeddings = parallel.parallel_apply(
                model_replicas, [((pcd, raw_pcd),) for pcd, raw_pcd in zip(batch_data['pcd'], batch_data['raw_pcd'])], devices=[self.master_device])
            embeddings = parallel.gather(embeddings, self.master_device, dim=0)
        else:
            embeddings = self.model((batch_data['pcd'][0], batch_data['raw_pcd'][0]))
        return embeddings
