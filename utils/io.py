import h5py
import numpy as np
import pyexr
import open3d
import os
import sys


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.npy':
            return cls._read_npy(file_path)
        elif file_extension == '.exr':
            return cls._read_exr(file_path)
        elif file_extension == '.pcd':
            return cls._read_pcd(file_path)
        elif file_extension == '.h5':
            return cls._read_h5(file_path)
        elif file_extension == '.txt':
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension == '.pcd':
            return cls._write_pcd(file_path, file_content)
        elif file_extension == '.h5':
            return cls._write_h5(file_path, file_content)
        elif file_extension == '.npy':
            return cls._write_npy(file_path, file_content)
        elif file_extension == '.ply':
            return cls._write_ply(file_path, file_content)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path).astype(np.float32)

    @classmethod
    def _read_exr(cls, file_path):
        return 1.0 / pyexr.open(file_path).get("Depth.Z").astype(np.float32)

    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        pc = np.asarray(pc.points, dtype=np.float32)
        return pc

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path)
        pc = np.asarray(f['data'][()], dtype=np.float32)
        return pc

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=file_content)

    @classmethod
    def _write_npy(cls, file_path, file_content):
        np.save(file_path, file_content)

    @classmethod
    def _write_ply(cls, file_path, file_content):
        tmp_file_path = file_path.replace('.ply', '.pcd')
        cls._write_pcd(tmp_file_path, file_content)
        _ = os.popen('/usr/local/bin/pcl_pcd2ply %s %s' %
                     (tmp_file_path, file_path))
        _ = os.popen('rm %s' % tmp_file_path)
