import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as oe


def binary_encode(arr):
    # encode input data as binary
    out = np.ndarray((arr.shape[0], 1), dtype=np.object)
    col_tags = dict()
    col_idx = []
    for col in range(arr.shape[1]):
        enc = le()
        arr[:, col] = enc.fit_transform(arr[:, col])
        col_tags[len(col_idx)] = enc.inverse_transform(np.unique(arr[:, col].astype(int)).tolist())
        o_enc = oe()
        tmp_ = o_enc.fit_transform(arr[:, col].reshape(-1, 1)).toarray()
        out = np.concatenate((out, tmp_), axis=1)
        col_idx.append(tmp_.shape[1])

    out = np.delete(out, 0, axis=1)
    return out, col_idx, col_tags


def load(path):
    out = pd.read_csv(path)
    out = out.dropna(axis=0, how='any')
    columns = out.columns.tolist()
    return out.to_numpy(), columns


def col_merge(data, idx, tags, nums=[]):
    # recover data from binary(or mixed) to its original format
    cur = 0
    out = np.zeros((data.shape[0], 1))
    for i, col_idx in enumerate(idx):
        tmp = data[:, cur:cur + col_idx]
        if col_idx != 1:
            tmp = np.argmax(tmp, axis=1)
            indx = tmp.tolist()
            tmp = np.array(tags[i])[indx].reshape(tmp.shape)
        else:
            tmp = tmp * (np.max(nums) - np.min(nums)) + np.min(nums)
        out = np.concatenate((out, tmp.reshape(-1, 1)), axis=1)
        cur += col_idx
    out = np.delete(out, 0, axis=1)

    return out