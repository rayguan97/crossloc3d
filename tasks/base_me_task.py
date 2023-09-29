import torch
import MinkowskiEngine as ME
import math
from abc import *


class BaseMETask(metaclass=ABCMeta):
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.log.info('Building %s task ...' % cfg.task_type)
        self.model = self._init_model()
        self.log.info('Parameters in %s task : %d' %
                      (cfg.task_type, self._count_model_params(self.model)))
        self.task_state = 'train'
        self.current_epoch = 0
        self.num_devices = 0
        self.devices = 0
        self.master_device = None

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError('You must implement _init_model method')

    def train(self):
        self.task_state = 'train'
        self.model.train()

    def eval(self):
        self.task_state = 'eval'
        self.model.eval()

    def cuda(self):
        self.num_devices = torch.cuda.device_count()
        self.devices = list(range(self.num_devices))
        self.master_device = self.devices[0]
        self.model = self.model.to(self.master_device)
        self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(
            self.model)

    def load(self, load_dir):
        self.log.info('Loading task state from %s ...' % load_dir)
        state = torch.load(load_dir)
        if 'model' in self.cfg.resume_items:
            self.model.load_state_dict(state['model_state_dict'], strict=False)
        if 'epoch' in self.cfg.resume_items:
            self.current_epoch = state['epoch']
        return state

    def save(self, save_dir, **kwargs):
        state = dict(kwargs)
        state['model_state_dict'] = self.model.state_dict()
        state['epoch'] = self.current_epoch
        if not self.cfg.debug:
            self.log.info('Saving task state to %s ...' % save_dir)
            torch.save(state, save_dir)

    def model_params(self):
        return self.model.parameters()

    def _count_model_params(self, model):
        return sum(p.numel() for p in model.parameters())

    @abstractmethod
    def _train_step(self, meta_data, batch_data, **kwargs):
        raise NotImplementedError('You must implement _train_step method')

    @abstractmethod
    def _eval_step(self, meta_data, batch_data, **kwargs):
        raise NotImplementedError('You must implement _eval_step method')

    def step(self, meta_data, batch_data, **kwargs):

        bs_per_device = math.ceil(
            len(batch_data['pcd_feats'])/self.num_devices)

        if self.task_state == 'eval':
            bs_per_device = len(batch_data['pcd_feats'])

        pcd_on_devices = []
        raw_pcd_on_devices = []

        for i in range(self.num_devices if self.task_state == 'train' else 1):
            coords = batch_data['pcd_coords'][i*bs_per_device:min(
                (i+1)*bs_per_device, len(batch_data['pcd_coords']))]
            coords = ME.utils.batched_coordinates(coords)
            feats = batch_data['pcd_feats'][i*bs_per_device:min(
                (i+1)*bs_per_device, len(batch_data['pcd_feats']))]
            feats = torch.cat(feats, dim=0)
            raw_pcd = torch.stack(batch_data['pcd'][i*bs_per_device:min(
                (i+1)*bs_per_device, len(batch_data['pcd']))], dim=0)

            with torch.cuda.device(self.devices[i]):
                pcd_on_devices.append(ME.SparseTensor(
                    feats.to(self.devices[i]), coordinates=coords.to(self.devices[i])))
                raw_pcd_on_devices.append(raw_pcd.to(self.devices[i]))

        batch_data['pcd'] = pcd_on_devices
        batch_data['raw_pcd'] = raw_pcd_on_devices
        del batch_data['pcd_feats']
        del batch_data['pcd_coords']

        if self.task_state == 'train':
            return self._train_step(meta_data, batch_data, **kwargs)
        elif self.task_state == 'eval':
            return self._eval_step(meta_data, batch_data, **kwargs)
        else:
            raise Exception('Unsupported task state: %s' % self.task_state)

    def before_epoch(self, epoch):
        self.current_epoch = epoch

    def after_epoch(self, epoch):
        self.current_epoch = epoch
