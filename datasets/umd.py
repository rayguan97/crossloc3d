import pickle
from tqdm import tqdm
from bitarray import bitarray
import os

from .utils.transformers import Compose
from .utils.dataset import TrainDataset, EvalDataset


LST = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20]
# LST = [1,2,3,4,5,6,7]
ACC_LST = ["umcp_lidar5_cloud_{}".format(i) for i in LST]

class UMDDataset(object):

    def __init__(self, cfg, log, debug=False, accept_lst=ACC_LST):
        self.cfg = cfg
        self.log = log
        self.debug = debug
        self.accept_lst = accept_lst

    def preprocess(self):
        catalog_file_path = self.cfg.train_catalog_file_path
        cached_catalog_file_path = self.cfg.cached_train_catalog_file_path
        if not os.path.exists(cached_catalog_file_path):
            self.log.info('Generating cache files for %s ...' %
                          catalog_file_path)
            with open(catalog_file_path, 'rb') as f:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                catalog = pickle.load(f)
            # Convert to bitarray
            newcatalog = dict()
            for ndx in tqdm(catalog.keys()):

                newcatalog[ndx] = catalog[ndx]
                newcatalog[ndx]['positives'] = set(catalog[ndx]['positives'])
                newcatalog[ndx]['negatives'] = set(catalog[ndx]['negatives'])
                pos_mask = [e_ndx in catalog[ndx]['positives']
                            for e_ndx in range(len(catalog))]
                neg_mask = [e_ndx in catalog[ndx]['negatives']
                            for e_ndx in range(len(catalog))]
                newcatalog[ndx]['positives'] = bitarray(pos_mask)
                newcatalog[ndx]['negatives'] = bitarray(neg_mask)

            catalog = newcatalog
            if not self.debug:
                with open(cached_catalog_file_path, 'wb') as f:
                    pickle.dump(catalog, f)
        else:
            with open(cached_catalog_file_path, 'rb') as f:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                catalog = pickle.load(f)
        return catalog

    def get_catalog(self, subset_type):
        if subset_type == 'train':
            return self.preprocess()
        else:
            with open(self.cfg[subset_type+'_file_path'], 'rb') as f:
                catalog = pickle.load(f)
            return catalog

    def subset(self, subset_type):
        assert subset_type in ['train', 'database', 'queries']
        transform = Compose(self.cfg.transform_cfg[subset_type])
        batch_transform = Compose(self.cfg.batch_transform_cfg[subset_type])

        if subset_type == 'train':
            return TrainDataset(self.cfg.data_root_dir, self.get_catalog(subset_type), transform, batch_transform, self.log, self.debug)
        else:
            return EvalDataset(self.cfg.data_root_dir, self.get_catalog(subset_type), transform, batch_transform, self.log, self.debug)
