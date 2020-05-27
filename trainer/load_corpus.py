import pickle as pkl

import numpy as np
import scipy as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

from trainer.prepare_matrices import normalize_adj

"""
Loads input corpus from gcn/data directory

ind.data_name.x => the feature
 vectors of the training docs as scipy.sparse.csr.csr_matrix object;
ind.data_name.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
ind.data_name.allx => the feature vectors of both labeled and unlabeled training docs/words
    (a superset of ind.data_name.x) as scipy.sparse.csr.csr_matrix object;
ind.data_name.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
ind.data_name.ty => the one-hot labels of the test docs as numpy.ndarray object;
ind.data_name.ally => the labels for instances in ind.data_name.allx as numpy.ndarray object;
ind.data_name.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
data_name.train => the indices of training docs in original doc list.

All objects above must be saved using python pickle module.
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, row_length: int):
    """Create mask."""
    mask = np.zeros(row_length)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)
    laplacian = sp.sparse.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.sparse.eye(adj.shape[0])

    t_k = list()
    # t_k.append(sp.eye(adj.shape[0]))
    # t_k.append(scaled_laplacian)
    t_k.append(sp.sparse.eye(adj.shape[0]).A)
    t_k.append(scaled_laplacian.A)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.sparse.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    # return sparse_to_tuple(t_k)
    return t_k


def load_corpus(data_name, split_index_dir: str, node_features_dir: str, adjacency_dir: str):
    """Loads all data input files (as well the training/test data)."""
    node_feature_names = ['x', 'y', 'tx', 'ty', 'allx', 'ally']
    node_features = []
    for node_feature_name in node_feature_names:
        with open(node_features_dir + "{}/ind.{}.{}".format(data_name, data_name, node_feature_name), 'rb') as f:
            node_features.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally = tuple(node_features)

    with open(adjacency_dir + "ind.{}.adj".format(data_name), 'rb') as f:
        adj = pkl.load(f, encoding='latin1')

    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.sparse.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    # print(len(labels))

    train_idx_orig = parse_index_file(split_index_dir + "{}.train".format(data_name))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size
