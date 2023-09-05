import numpy as np


def pcal(x, columns, col_idx):
    # Compute ppp
    # Since the comparison is done with an AND operation, the input data must be binary.
    if np.max(x) > 1:
        raise ValueError("Input has to be encoded in one-hot vector")
    return out_loop(x, len(columns), col_idx)


def out_loop(x, n_cols, col_idx):
    """
    Column wised loop
    :param x: (ndarray) input
    :param n_cols: (int) number of columns
    :param col_idx: (list) list of max value for each column
    :return: (float) PPP
    """
    rows = x.shape[0]
    cols = n_cols
    out = np.empty((rows, cols))
    s_i = in_loop(x, cols).reshape((rows,))
    for col in range(cols):
        if col == 0:
            tmp = x[:, col_idx[col]:].copy()
        elif col + 1 == n_cols:
            tmp = x[:, :sum(col_idx[:col])].copy()
        else:
            tmp = np.concatenate((x[:, :sum(col_idx[:col])], x[:, sum(col_idx[:col+1]):]), 1)
        out[:, col] = s_i/in_loop(tmp, cols-1).reshape((rows,))
    return 1 - out


def in_loop(x, n_cols):
    # compare one row to another
    out = np.empty((x.shape[0], 1))
    for row in range(x.shape[0]):
        identical = np.floor(np.matmul(x, x[row])/n_cols).sum()
        out[row] = identical
    return out
