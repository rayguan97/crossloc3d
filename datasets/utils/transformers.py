import numpy as np
import transforms3d
import torch
import math
from scipy.linalg import expm, norm
import random


class BaseTransformer(object):
    def __init__(self, params, **kwargs):
        self.params = params
        self.add_params(**kwargs)

    def add_params(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.params:
                self.params[k] = v
        if 'group' not in self.params:
            self.params['group'] = False
        if 'batch' not in self.params:
            self.params['batch'] = False

    def __call__(self, data, objs):
        if self.params['group']:
            return self._group_call(data, objs)
        elif self.params['batch']:
            return self._batch_call(data, objs)
        else:
            return self._sep_call(data, objs)


class ToTensor(BaseTransformer):

    def _sep_call(self, data, objs):
        for k, v in list(data.items()):
            if k not in objs:
                continue
            data[k] = torch.from_numpy(v)
        return data

    def _group_call(self, data, objs):
        return self._sep_call(data, objs)


class DownSample3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='random')
        assert 'npts' in self.params

    def _sep_call(self, data, objs):
        if self.params['method'] == 'random':
            for k, v in data.items():
                if k not in objs:
                    continue
                npts = v.shape[0]
                indices = np.random.permutation(npts)[:self.params['npts']]
                data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'random':
            npts = None
            for k, v in data.items():
                if k not in objs:
                    continue
                if npts is None:
                    npts = v.shape[0]
                else:
                    assert npts == v.shape[0]
            indices = np.random.permutation(npts)[:self.params['npts']]
            for k, v in list(data.items()):
                if k in objs:
                    data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))

        return data


class UpSample3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='replicate')
        assert 'npts' in self.params

    def _sep_call(self, data, objs):
        if self.params['method'] == 'fill':
            for k, v in list(data.items()):
                if k in objs and v.shape[0] < self.params['npts']:
                    if len(v.shape) == 1:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]]) * self.params['fill_value']
                    else:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]] + list(v.shape[1:]))*self.params['fill_value']
                    fill_values = fill_values.astype(v.dtype)
                    data[k] = np.concatenate((data[k], fill_values), axis=0)
        elif self.params['method'] == 'replicate':
            for k, v in data.items():
                if k not in objs:
                    continue
                npts = v.shape[0]
                num_copies = math.ceil(self.params['npts'] / npts)
                indices = np.random.permutation(npts)
                data[k] = np.tile(v[indices], [num_copies]+[1]*(len(v.shape)-1))[
                    : self.params['npts']]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'fill':
            for k, v in list(data.items()):
                if k in objs and v.shape[0] < self.params['npts']:
                    if len(v.shape) == 1:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]]) * self.params['fill_value']
                    else:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]] + v.shape[1:])*self.params['fill_value']
                    fill_values = fill_values.astype(v.dtype)
                    data[k] = np.concatenate((data[k], fill_values), axis=0)
        elif self.params['method'] == 'replicate':
            npts = None
            for k, v in data.items():
                if k not in objs:
                    continue
                if npts is None:
                    npts = v.shape[0]
                else:
                    assert npts == v.shape[0]
            num_copies = math.ceil(self.params['npts'] / npts)
            indices = np.random.permutation(npts)
            for k, v in list(data.items()):
                if k in objs:
                    data[k] = np.tile(v[indices], [num_copies]+[1]*(len(v.shape)-1))[
                        : self.params['npts']]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Sample3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='replicate')
        assert 'npts' in self.params

    def _sep_call(self, data, objs):
        if self.params['method'] == 'fill':
            for k, v in list(data.items()):
                if k in objs and v.shape[0] < self.params['npts']:
                    if len(v.shape) == 1:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]]) * self.params['fill_value']
                    else:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]] + list(v.shape[1:]))*self.params['fill_value']
                    fill_values = fill_values.astype(v.dtype)
                    data[k] = np.concatenate((data[k], fill_values), axis=0)
                elif k in objs and v.shape[0] > self.params['npts']:
                    npts = v.shape[0]
                    indices = np.random.permutation(npts)[:self.params['npts']]
                    data[k] = v[indices]
        elif self.params['method'] == 'replicate':
            for k, v in data.items():
                if k not in objs:
                    continue
                if v.shape[0] < self.params['npts']:
                    npts = v.shape[0]
                    num_copies = math.ceil(self.params['npts'] / npts)
                    indices = np.random.permutation(npts)
                    data[k] = np.tile(v[indices], [num_copies]+[1]*(len(v.shape)-1))[
                        : self.params['npts']]
                elif v.shape[0] > self.params['npts']:
                    npts = v.shape[0]
                    indices = np.random.permutation(npts)[:self.params['npts']]
                    data[k] = v[indices]

        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'fill':
            for k, v in list(data.items()):
                if k in objs and v.shape[0] < self.params['npts']:
                    if len(v.shape) == 1:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]]) * self.params['fill_value']
                    else:
                        fill_values = np.ones(
                            [self.params['npts'] - v.shape[0]] + v.shape[1:])*self.params['fill_value']
                    fill_values = fill_values.astype(v.dtype)
                    data[k] = np.concatenate((data[k], fill_values), axis=0)
        elif self.params['method'] == 'replicate':
            npts = None
            for k, v in data.items():
                if k not in objs:
                    continue
                if npts is None:
                    npts = v.shape[0]
                else:
                    assert npts == v.shape[0]
            num_copies = math.ceil(self.params['npts'] / npts)
            indices = np.random.permutation(npts)
            for k, v in list(data.items()):
                if k in objs:
                    data[k] = np.tile(v[indices], [num_copies]+[1]*(len(v.shape)-1))[
                        : self.params['npts']]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Permute3D(BaseTransformer):

    def __init__(self, params):
        super().__init__(params, method='random')

    def _sep_call(self, data, objs):
        if self.params['method'] == 'random':
            for k, v in data.items():
                if k not in objs:
                    continue
                npts = v.shape[0]
                indices = np.random.permutation(npts)
                data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'random':
            npts = None
            for k, v in data.items():
                if k not in objs:
                    continue
                if npts is None:
                    npts = v.shape[0]
                else:
                    assert npts == v.shape[0]
            indices = np.random.permutation(npts)
            for k, v in list(data.items()):
                if k in objs:
                    data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Scale3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='random')

    def _sep_call(self, data, objs):
        if self.params['method'] == 'random':
            for k, v in data.items():
                if k not in objs:
                    continue
                scale = np.random.uniform(
                    1.0/self.params['scale'], self.params['scale'])
                trfm_mat = transforms3d.zooms.zfdir2mat(scale)
                data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        elif self.params['method'] == 'fixed':
            scale = self.params['scale']
            trfm_mat = transforms3d.zooms.zfdir2mat(scale)
            for k, v in data.items():
                if k in objs:
                    data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'random':
            scale = np.random.uniform(
                1.0/self.params['scale'], self.params['scale'])
            trfm_mat = transforms3d.zooms.zfdir2mat(scale)
            for k, v in data.items():
                if k in objs:
                    data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        elif self.params['method'] == 'fixed':
            scale = self.params['scale']
            trfm_mat = transforms3d.zooms.zfdir2mat(scale)
            for k, v in data.items():
                if k in objs:
                    data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Mirror3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='xz_plane')
        if self.params['method'] == 'xyz_plane':
            self.add_params(p=[0.33, 0.33, 0.33])

    def _sep_call(self, data, objs):
        if self.params['method'] == 'xz_plane':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)

            for k, v in list(data.items()):
                if k not in objs:
                    continue
                rnd_value = np.random.random()
                if rnd_value <= 0.25:
                    trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                    trfm_mat = np.dot(trfm_mat_z, trfm_mat)
                elif rnd_value > 0.25 and rnd_value <= 0.5:
                    trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                elif rnd_value > 0.5 and rnd_value <= 0.75:
                    trfm_mat = np.dot(trfm_mat_z, trfm_mat)
                data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        elif self.params['method'] == 'xyz_plane':
            p_cum_sum = np.cumsum(self.params['p'])
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                r = random.random()
                if r <= p_cum_sum[0]:
                    data[k][:, 0] = -v[:, 0]
                elif r <= p_cum_sum[1]:
                    data[k][:, 1] = -v[:, 1]
                elif r <= p_cum_sum[2]:
                    data[k][:, 2] = -v[:, 2]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'xz_plane':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            rnd_value = np.random.random()
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            for k, v in list(data.items()):
                if k in objs:
                    data[k][:, :3] = np.dot(v[:, :3], trfm_mat.T)
        elif self.params['method'] == 'xyz_plane':
            p_cum_sum = np.cumsum(self.params['p'])
            r = random.random()
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                if r <= p_cum_sum[0]:
                    data[k][:, 0] = -v[:, 0]
                elif r <= p_cum_sum[1]:
                    data[k][:, 1] = -v[:, 1]
                elif r <= p_cum_sum[2]:
                    data[k][:, 2] = -v[:, 2]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _batch_call(self, data, objs):

        if self.params['method'] == 'xyz_plane':
            p_cum_sum = np.cumsum(self.params['p'])
            r = random.random()
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                if r <= p_cum_sum[0]:
                    data[k][..., 0] = -v[..., 0]
                elif r <= p_cum_sum[1]:
                    data[k][..., 1] = -v[..., 1]
                elif r <= p_cum_sum[2]:
                    data[k][..., 2] = -v[..., 2]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Rotation3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, axis=np.random.rand(
            3) - 0.5, max_theta1=180, max_theta2=15)

    def _M(self, axis, theta):
        axis = np.array(axis, dtype=np.float32)
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def _sep_call(self, data, objs):

        for k, v in list(data.items()):
            if k not in objs:
                continue
            R = self._M(self.params['axis'], (np.pi *
                                              self.params['max_theta1'] / 180) * 2 * (np.random.rand(1) - 0.5))
            if self.params['max_theta2'] is not None:
                R_n = self._M(np.random.rand(
                    3) - 0.5, (np.pi * self.params['max_theta2'] / 180) * 2 * (np.random.rand(1) - 0.5))
                R = R @ R_n
            data[k][:, :3] = np.dot(v[:, :3], R)

        return data

    def _group_call(self, data, objs):
        R = self._M(self.params['axis'], (np.pi *
                                          self.params['max_theta1'] / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.params['max_theta2'] is not None:
            R_n = self._M(np.random.rand(
                3) - 0.5, (np.pi * self.params['max_theta2'] / 180) * 2 * (np.random.rand(1) - 0.5))
            R = R @ R_n

        for k, v in list(data.items()):
            if k in objs:
                data[k][:, :3] = np.dot(v[:, :3], R)

        return data

    def _batch_call(self, data, objs):
        R = self._M(self.params['axis'], (np.pi *
                                          self.params['max_theta1'] / 180) * 2 * (np.random.rand(1) - 0.5))
        if self.params['max_theta2'] is not None:
            R_n = self._M(np.random.rand(
                3) - 0.5, (np.pi * self.params['max_theta2'] / 180) * 2 * (np.random.rand(1) - 0.5))
            R = R @ R_n

        for k, v in list(data.items()):
            if k in objs:
                data[k][..., :3] = v[..., :3] @ torch.tensor(R, dtype=v.dtype)

        return data


class Translation3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, max_delta=0.05)

    def _sep_call(self, data, objs):
        for k, v in list(data.items()):
            if k not in objs:
                continue
            T = self.params['max_delta'] * np.random.randn(1, 3)
            data[k][:, :3] += T.astype(v.dtype)
        return data

    def _group_call(self, data, objs):
        T = self.params['max_delta'] * np.random.randn(1, 3)
        for k, v in list(data.items()):
            if k in objs:
                data[k][:, :3] += T.astype(v.dtype)
        return data


class Shear3D(BaseTransformer):

    def __init__(self, params):
        super().__init__(params, max_delta=0.1)

    def _sep_call(self, data, objs):
        for k, v in list(data.items()):
            if k not in objs:
                continue
            T = np.eye(3) + self.params['max_delta'] * np.random.randn(3, 3)
            data[k][:, :3] = v[:, :3] @ T.astype(v.dtype)
        return data

    def _group_call(self, data, objs):
        T = np.eye(3) + self.params['max_delta'] * np.random.randn(3, 3)
        for k, v in list(data.items()):
            if k in objs:
                data[k][:, :3] = v[:, :3] @ T.astype(v.dtype)
        return data


class Jitter3D(BaseTransformer):

    def __init__(self, params):
        super().__init__(params, sigma=0.01, clip=None, p=1.)
        assert 0 < self.params['p'] <= 1.
        assert self.params['sigma'] > 0.

    def _sep_call(self, data, objs):
        for k, v in data.items():
            if k not in objs:
                continue
            npts = v.shape[0]
            mask = np.random.randn(npts) < self.params['p']
            jitter = self.params['sigma'] * \
                np.random.randn(*v[mask][:3].shape)
            if self.params['clip'] is not None:
                jitter = np.clip(
                    jitter, -self.params['clip'], self.params['clip'])
            data[k][mask][:3] = v[mask][:3]+jitter
        return data

    def _group_call(self, data, objs):
        npts = None
        for k, v in data.items():
            if k not in objs:
                continue
            if npts is None:
                npts = v.shape[0]
            else:
                assert npts == v.shape[0]

        mask = np.random.randn(npts) < self.params['p']

        for k, v in list(data.items()):
            if k in objs:
                jitter = self.params['sigma'] * \
                    np.random.randn(v[mask][:3].shape)
                if self.params['clip'] is not None:
                    jitter = np.clip(
                        jitter, -self.params['clip'], self.params['clip'])
                data[k][mask][:3] = v[mask][:3]+jitter
        return data


class Drop3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='random')
        if self.params['method'] == 'random':
            self.add_params(min_dr=0.0, max_dr=0.5)
        if self.params['method'] == 'cuboid':
            self.add_params(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            assert 'bbox_pc_name' in self.params

    def _get_cuboid(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords
        min_coords = np.min(flattened_coords, axis=0)
        max_coords = np.max(flattened_coords, axis=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(
            self.params['scale'][0], self.params['scale'][1]) * area
        aspect_ratio = random.uniform(
            self.params['ratio'][0], self.params['ratio'][1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def _sep_call(self, data, objs):
        if self.params['method'] == 'random':
            for k, v in data.items():
                if k not in objs:
                    continue
                npts = v.shape[0]
                dr = random.uniform(
                    self.params['min_dr'], self.params['max_dr'])
                mask = np.random.choice(
                    range(npts), size=int(npts*dr), replace=False)
                data[k][mask] = np.zeros_like(v[mask])
        elif self.params['method'] == 'cuboid':
            for k, v in data.items():
                if k not in objs:
                    continue
                if random.random() < self.params['p']:
                    coords = v[:, :3]
                    npts = coords.shape[0]
                    x, y, w, h = self._get_cuboid(coords)
                    mask = (x < coords[..., 0]) & (
                        coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
                    data[k][mask] = np.zeros_like(v[mask])
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'random':
            npts = None
            for k, v in data.items():
                if k not in objs:
                    continue
                if npts is None:
                    npts = v.shape[0]
                else:
                    assert npts == v.shape[0]
            dr = random.uniform(self.params['min_dr'], self.params['max_dr'])
            mask = np.random.choice(range(npts), size=int(n*r), replace=False)
            for k, v in data.items():
                if k in objs:
                    data[k][mask] = np.zeros_like(v[mask])
        elif self.params['method'] == 'cuboid':
            if random.random() < self.params['p']:

                coords = data[self.params['bbox_pc_name']]
                npts = coords.shape[0]
                x, y, w, h = self._get_params(coords)
                for k, v in data.items():
                    if k in objs:
                        assert npts == v.shape[0]
                mask = (x < coords[..., 0]) & (
                    coords[..., 0] < x+w) & (y < coords[..., 1]) & (coords[..., 1] < y+h)
                for k, v in data.items():
                    if k in objs:
                        data[k][mask] = np.zeros_like(v[mask])
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Crop3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='sphere')
        if self.params['group']:
            self.add_params(eps=0.0)

    def _sep_call(self, data, objs):
        if self.params['method'] == 'sphere':
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                center = v[random.randint(0, v.shape[0] - 1), :3]
                pc = v[:, :3]
                dis2 = np.sum((pc - center) ** 2, axis=1)
                indices = dis2 <= self.params['radius']
                data[k] = v[indices]
        elif self.params['method'] == 'knn':
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                center = v[random.randint(0, v.shape[0] - 1), :3]
                pc = v[:, :3]
                dis2 = np.sum((pc - center) ** 2, axis=1)
                indices = np.argsort(dis2)[:self.params['npts']]
                data[k] = v[indices]
        elif self.params['method'] == 'knn_angle':
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                pc = v[:, :3]
                norm = np.random.normal(size=(1, 3))
                norm /= np.linalg.norm(norm, ord=2, axis=1, keepdims=True)
                dis = np.squeeze(pc @ norm.T)
                indices = np.argsort(dis)[:self.params['npts']]
                data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        if self.params['method'] == 'sphere':
            center = None
            for k, v in list(data.items()):
                if k == self.params['center_pc_name']:
                    center = v[random.randint(0, v.shape[0] - 1), :3]
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                pc = v[:, :3]
                c = center + (np.random.rand(3)-0.5)*2*self.params['eps']
                dis2 = np.sum((pc - c) ** 2, axis=1)
                indices = dis2 <= self.params['radius']
                data[k] = v[indices]
        elif self.params['method'] == 'knn':
            center = None
            for k, v in list(data.items()):
                if k == self.params['center_pc_name']:
                    center = v[random.randint(0, v.shape[0] - 1), :3]
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                pc = v[:, :3]
                c = center + (np.random.rand(3) - 0.5) * 2 * self.params['eps']
                dis2 = np.sum((pc - c) ** 2, axis=1)
                indices = np.argsort(dis2)[:self.params['npts']]
                # assert np.all(np.sort(indices) == np.arange(v.shape[0]))
                data[k] = v[indices]
        elif self.params['method'] == 'knn_angle':
            norm = np.random.normal(size=(1, 3))
            norm /= np.linalg.norm(norm, ord=2, axis=1, keepdims=True)
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                pc = v[:, :3]
                norm2 = np.random.normal(size=(1, 3))
                norm2 = norm2 - np.sum(norm * norm2) * norm
                norm2 /= np.linalg.norm(norm2, ord=2, axis=1, keepdims=True)
                assert np.sum(norm * norm2) < 1e-6
                norm_tr = norm + norm2 * self.params['eps']
                norm_tr /= np.linalg.norm(norm_tr,
                                          ord=2, axis=1, keepdims=True)
                dis = np.squeeze(pc @ norm_tr.T)
                indices = np.argsort(dis)[:self.params['npts']]
                data[k] = v[indices]
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data


class Normalize3D(BaseTransformer):
    def __init__(self, params):
        super().__init__(params, method='cube')

    def _sep_call(self, data, objs):
        if self.params['method'] == 'cube':
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                norm_pc = v[:, :3] - np.mean(v[:, :3], axis=0)
                # npts, 3
                mean_dis = np.max(np.mean(np.abs(norm_pc), axis=0))
                norm_pc /= mean_dis
                data[k][:, :3] = norm_pc
        elif self.params['method'] == 'sphere':
            for k, v in list(data.items()):
                if k not in objs:
                    continue
                norm_pc = v[:, :3] - np.mean(v[:, :3], axis=0)
                # npts, 3
                mean_dis = np.mean(np.linalg.norm(norm_pc, ord=2, axis=1))
                norm_pc /= mean_dis
                data[k][:, :3] = norm_pc
        else:
            raise NotImplementedError(
                '{} has not been implemented!'.format(self.params['method']))
        return data

    def _group_call(self, data, objs):
        raise NotImplementedError(
            'Normalize _group_call has not been implemented!')


class Compose(object):
    def __init__(self, trfm_cfg):
        self.transformers = []
        for t_cfg in trfm_cfg:
            transformer = eval(t_cfg['type'])
            params = t_cfg['params'] if 'params' in t_cfg else {}
            self.transformers.append({
                'callback': transformer(params),
                'objs': t_cfg['objs']
            })

    def __call__(self, data):
        for t in self.transformers:
            transformer = t['callback']
            objs = t['objs']
            data = transformer(data, objs)
        return data
