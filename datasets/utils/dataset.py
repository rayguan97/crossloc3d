import os
import pickle
import numpy as np
import math
import random
import torch
from bitarray import bitarray


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, data_root_dir, catalog, transform=None, batch_transform=None, log=None, debug=False):
        self.catalog = catalog
        self.transform = transform
        self.batch_transform = batch_transform
        self.data_root_dir = data_root_dir
        self.debug = debug
        self.log = log

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.catalog[ndx]['query']
        pcd = self.load_pc(filename).astype(np.float32)
        # meta = {'idx': ndx, 'filename': filename, 'n': self.catalog[ndx]['n'], 'e':self.catalog[ndx]['e']}
        meta = {'idx': ndx, 'filename': filename}
        if "northing" in self.catalog[ndx].keys():
            meta["northing"] = self.catalog[ndx]["northing"]

        if "easting" in self.catalog[ndx].keys():
            meta["easting"] = self.catalog[ndx]["easting"]

        data = {'pcd': pcd}
        if self.transform:
            data = self.transform(data)
        return meta, data

    def get_pos_pairs(self, ndx):
        # Get list of indexes of similar clouds
        return self.catalog[ndx]['positives'].search(bitarray([True]))

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.data_root_dir, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == 4096 * \
            3, "Error in point cloud shape: {}".format(filename)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        return pc


class EvalDataset(torch.utils.data.Dataset):

    def __init__(self, data_root_dir, catalog, transform=None, batch_transform=None, log=None, debug=False):
        self.catalog = catalog
        self.transform = transform
        self.batch_transform = batch_transform
        self.data_root_dir = data_root_dir
        self.debug = debug
        self.subset_num = 0
        self.log = log

    def __len__(self):
        return len(self.catalog[self.subset_num])

    def set_subset(self, idx):
        self.subset_num = idx

    def subset_len(self):
        return len(self.catalog)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.catalog[self.subset_num][ndx]['query']
        pcd = self.load_pc(filename).astype(np.float32)
        meta = {'idx': ndx, 'filename': filename}
        if "northing" in self.catalog[self.subset_num][ndx].keys():
            meta["northing"] = self.catalog[self.subset_num][ndx]["northing"]

        if "easting" in self.catalog[self.subset_num][ndx].keys():
            meta["easting"] = self.catalog[self.subset_num][ndx]["easting"]

        data = {'pcd': pcd}
        if self.transform:
            data = self.transform(data)
        return meta, data

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.data_root_dir, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == 4096 * \
            3, "Error in point cloud shape: {}".format(filename)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        return pc
