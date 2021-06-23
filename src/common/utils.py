import h5py
import torch


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def to_onehot(indexes, num, dtype=None, numpy=False):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,), dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot
