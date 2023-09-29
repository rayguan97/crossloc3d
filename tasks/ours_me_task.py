from .base_me_task import BaseMETask
from losses import create_loss_fn
from models import create_model
import torch.nn.parallel as parallel
import MinkowskiEngine as ME
import math
import torch


class OursMETask(BaseMETask):
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

        # from IPython import embed;embed()

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

    def step(self, meta_data, batch_data, **kwargs):

        bs_per_device = math.ceil(
            len(batch_data['pcd'])/self.num_devices)

        if self.task_state == 'eval':
            bs_per_device = len(batch_data['pcd'])

        pcd_on_devices = []
        raw_pcd_on_devices = []

        coords_lst = []
        feats_lst = []
        k_lst = []

        for i in range(self.num_devices if self.task_state == 'train' else 1):

            for k in batch_data.keys():
                if "pcd_coords" in k:
                    coords = batch_data[k][i*bs_per_device:min(
                        (i+1)*bs_per_device, len(batch_data[k]))]
                    coords = ME.utils.batched_coordinates(coords)
                    coords_lst.append(coords)
                    if not k in k_lst:
                        k_lst.append(k) 
                if "pcd_feats" in k:
                    feats = batch_data[k][i*bs_per_device:min(
                        (i+1)*bs_per_device, len(batch_data[k]))]
                    feats = torch.cat(feats, dim=0)
                    feats_lst.append(feats)
                    if not k in k_lst:
                        k_lst.append(k) 
                        
            raw_pcd = torch.stack(batch_data['pcd'][i*bs_per_device:min(
                (i+1)*bs_per_device, len(batch_data['pcd']))], dim=0)

            with torch.cuda.device(self.devices[i]):
                pcd_on_devices.append([ME.SparseTensor(
                    feats_lst[c].to(self.devices[i]), coordinates=coords_lst[c].to(self.devices[i])) 
                    for c in range(len(coords_lst))])
                raw_pcd_on_devices.append(raw_pcd.to(self.devices[i]))

        batch_data['pcd'] = pcd_on_devices
        batch_data['raw_pcd'] = raw_pcd_on_devices

        for k in k_lst:
            del batch_data[k]


        if self.task_state == 'train':
            return self._train_step(meta_data, batch_data, **kwargs)
        elif self.task_state == 'eval':
            return self._eval_step(meta_data, batch_data, **kwargs)
        else:
            raise Exception('Unsupported task state: %s' % self.task_state)
