import numpy as np
import os, torch
import lmdb
import msgpack_numpy
from torch.utils.data import Dataset


class fastMRIdatasetKnee(Dataset):
    def __init__(self, root_dir="data/fastMRI", train=True):

        if train:
            self.path = os.path.join(root_dir, 'knee_singlecoil_train/')
        else:
            self.path = os.path.join(root_dir, 'knee_singlecoil_val/')

        self._lmdb_file = self.path
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(self._lmdb_file, map_size=1 << 36, readonly=True, lock=False)
        with self._lmdb_env.begin(buffers=True) as txn:
            ele = np.array(msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False))
        ele = (ele - np.min(ele)) / (np.max(ele) - np.min(ele))
        return ele
