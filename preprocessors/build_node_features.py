import pickle as pkl
from collections import OrderedDict
from math import ceil
from typing import List, Dict, MutableMapping, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from common import check_data_set
from preprocessors.configs import PreProcessingConfigs
from utils.file_ops import check_paths, create_dir

WORD_VECTORS_TYPE = MutableMapping[str, np.ndarray]  # word -> word-vector, words are ordered with respect to vocabulary


def extract_doc_labels(ds_corpus_meta_file: str) -> List[str]:
    with open(ds_corpus_meta_file) as ds_corpus_meta:
        doc_labels = list(OrderedDict.fromkeys(doc_meta.split()[2] for doc_meta in ds_corpus_meta))
    return doc_labels


def compute_x(docs_of_words: List[List[str]], tr_size: int, emb_dim: int, w_vectors: WORD_VECTORS_TYPE) -> csr_matrix:
    """ x: feature vectors of training docs, no initial features """
    data_x = []
    for i in range(tr_size):
        doc_vec = np.zeros(emb_dim, dtype=float)  # Initialize
        words = docs_of_words[i]
        for word in words:
            if word in w_vectors:
                doc_vec += w_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_x.extend(mean_doc_vec)

    row_indexes = np.array([[i] * emb_dim for i in range(tr_size)]).flatten().tolist()
    col_indexes = list(range(emb_dim)) * tr_size
    return csr_matrix((data_x, (row_indexes, col_indexes)), shape=(tr_size, emb_dim))


def compute_y(doc_meta_list: List[str], train_size: int, doc_labels: List[str]) -> np.ndarray:
    y = []
    for i in range(train_size):
        doc_meta = doc_meta_list[i]
        one_hot_encoded_label = [0] * len(doc_labels)

        label = doc_meta.split('\t')[2]
        label_index = doc_labels.index(label)
        one_hot_encoded_label[label_index] = 1
        y.append(one_hot_encoded_label)
    return np.array(y)


def compute_tx(docs_of_words: List[List[str]], test_size: int, real_train_size: int, word_emb_dim: int,
               w_vectors: WORD_VECTORS_TYPE) -> csr_matrix:
    """ tx: feature vectors of test docs, no initial features """
    data_tx = []
    for i in range(test_size):
        doc_vec = np.zeros(word_emb_dim, dtype=float)  # Initialize
        words = docs_of_words[i + real_train_size]
        for word in words:
            if word in w_vectors:
                doc_vec += w_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_tx.extend(mean_doc_vec)

    row_indexes = np.array([[i] * word_emb_dim for i in range(test_size)]).flatten().tolist()
    col_indexes = list(range(word_emb_dim)) * test_size
    return csr_matrix((data_tx, (row_indexes, col_indexes)), shape=(test_size, word_emb_dim))


def compute_ty(doc_meta_list: List[str], test_size: int, real_train_size: int, doc_labels: List[str]) -> np.ndarray:
    ty = []
    for i in range(test_size):
        doc_meta = doc_meta_list[i + real_train_size]
        one_hot_encoded_label = [0] * len(doc_labels)

        label = doc_meta.split('\t')[2]
        label_index = doc_labels.index(label)
        one_hot_encoded_label[label_index] = 1
        ty.append(one_hot_encoded_label)
    return np.array(ty)


def compute_allx(docs_of_words: List[List[str]], real_train_size: int, vocab: List[str],
                 word_vectors: WORD_VECTORS_TYPE, emb_dim: int) -> csr_matrix:
    """allx: A superset of x, the feature vectors of both labeled and words (unlabeled training instances)"""
    word_vectors_arr = extract_word_vectors_arr(word_vectors, vocab, emb_dim=emb_dim)
    data_allx = []
    row_size = real_train_size + len(vocab)

    for i in range(real_train_size):
        doc_vec = np.zeros(emb_dim, dtype=float)  # Initialize
        words = docs_of_words[i]
        for word in words:
            if word in word_vectors:
                doc_vec += word_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_allx.extend(mean_doc_vec)

    data_allx.extend(word_vectors_arr.flatten())
    data_allx = np.array(data_allx)
    row_indexes = np.array([[i] * emb_dim for i in range(row_size)]).flatten()
    col_indexes = np.array(list(range(emb_dim)) * row_size)

    return csr_matrix((data_allx, (row_indexes, col_indexes)), shape=(row_size, emb_dim))


def compute_ally(doc_meta_list: List[str], real_train_size: int, doc_labels: List[str], vocab_size: int) -> np.ndarray:
    ally = []
    for doc_meta in doc_meta_list[:real_train_size]:
        label = doc_meta.split('\t')[2]
        one_hot_encoded_label = [0] * len(doc_labels)
        label_index = doc_labels.index(label)
        one_hot_encoded_label[label_index] = 1
        ally.append(one_hot_encoded_label)

    zero_filled_one_hot_for_words = [np.zeros(len(doc_labels), dtype=int)] * vocab_size
    ally.extend(zero_filled_one_hot_for_words)
    return np.array(ally)


def load_word_to_word_vectors(path: str) -> Tuple[WORD_VECTORS_TYPE, int]:
    word_vectors_as_list = pkl.load(file=open(path, 'rb'))  # type: MutableMapping[str,List[float]]
    word_vectors = OrderedDict((word, np.array(vec_lst)) for word, vec_lst in word_vectors_as_list.items())
    word_embedding_dimension = len(next(iter(word_vectors.values())))
    return word_vectors, word_embedding_dimension


def extract_word_vectors_arr(w_vectors: WORD_VECTORS_TYPE, vocab: List[str], emb_dim: int) -> np.ndarray:
    np.random.seed(0)  # For reproducibility
    word_vectors_arr = np.random.uniform(-0.01, 0.01, (len(vocab), emb_dim))
    if len(w_vectors) != 0:
        for i, word in enumerate(vocab):
            if word in w_vectors:
                word_vectors_arr[i] = w_vectors[word]
    return word_vectors_arr


def dump_node_features(directory: str, ds: str, node_features_dict: Dict[str, Union[np.ndarray, csr_matrix]]):
    for name, node_feature_matrix in node_features_dict.items():
        with open(directory + "/ind.{}.{}".format(ds, name), 'wb') as file:
            pkl.dump(node_feature_matrix, file)


def build_node_features(ds_name: str, validation_ratio: float, use_predefined_word_vectors: bool,
                        cfg: PreProcessingConfigs):
    # input files for building node features
    ds_corpus = cfg.corpus_shuffled_dir + ds_name + '.txt'
    ds_corpus_meta = cfg.corpus_shuffled_meta_dir + ds_name + '.meta'
    ds_corpus_vocabulary = cfg.corpus_shuffled_vocab_dir + ds_name + '.vocab'
    ds_corpus_train_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.train'
    ds_corpus_test_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.test'

    # output directory of node features
    dir_corpus_node_features = cfg.corpus_shuffled_node_features_dir + "/" + ds_name

    # checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(ds_corpus, ds_corpus_meta, ds_corpus_vocabulary)
    check_paths(ds_corpus_train_idx, ds_corpus_train_idx)

    # Create output directory of node features
    create_dir(dir_path=dir_corpus_node_features, overwrite=False)

    # Adjust train size, for different training rates, for example: use 90% of training set
    real_train_size = len(open(ds_corpus_train_idx).readlines())
    adjusted_train_size = ceil(real_train_size * (1.0 - validation_ratio))
    test_size = len(open(ds_corpus_test_idx).readlines())

    # Extract word_vectors and word_embedding_dimension
    if use_predefined_word_vectors:
        ds_corpus_word_vectors = cfg.corpus_shuffled_word_vectors_dir + ds_name + '.word_vectors'
        # ds_corpus_word_vectors =  'glove.6B.300d.txt'  # Alternatively, you can use GLOVE word-embeddings
        word_vectors, word_emb_dim = load_word_to_word_vectors(path=ds_corpus_word_vectors)
    else:
        word_vectors, word_emb_dim = OrderedDict(), 300  # todo: parametrize

    vocabulary = open(ds_corpus_vocabulary).read().splitlines()  # Extract Vocabulary
    doc_meta_list = open(file=ds_corpus_meta, mode='r').read().splitlines()  # Extract Meta List
    doc_labels = extract_doc_labels(ds_corpus_meta_file=ds_corpus_meta)  # Extract Document Labels

    docs_of_words = [line.split() for line in open(file=ds_corpus)]  # Extract Documents of Words

    # Extract mean document word vectors and one hot labels of train-set
    x = compute_x(docs_of_words, adjusted_train_size, word_emb_dim, w_vectors=word_vectors)
    y = compute_y(doc_meta_list, train_size=adjusted_train_size, doc_labels=doc_labels)

    # Extract mean document word vectors and one hot labels of test-set
    tx = compute_tx(docs_of_words, test_size, real_train_size, word_emb_dim, w_vectors=word_vectors)
    ty = compute_ty(doc_meta_list, test_size=test_size, real_train_size=real_train_size, doc_labels=doc_labels)

    # Extract doc_features + word_features
    allx = compute_allx(docs_of_words, real_train_size, vocabulary, word_vectors, emb_dim=word_emb_dim)
    ally = compute_ally(doc_meta_list, real_train_size, doc_labels, vocab_size=len(vocabulary))

    # Dump node features matrices to files
    node_feature_matrices = {"x": x, "y": y, "tx": tx, "ty": ty, "allx": allx, "ally": ally}
    dump_node_features(directory=dir_corpus_node_features, ds=ds_name, node_features_dict=node_feature_matrices)

    print("[INFO] x.shape=   {},\t y.shape=   {}".format(x.shape, y.shape))
    print("[INFO] tx.shape=  {},\t ty.shape=  {}".format(tx.shape, ty.shape))
    print("[INFO] allx.shape={},\t ally.shape={}".format(allx.shape, ally.shape))
    print("[INFO] ========= EXTRACTED NODE FEATURES: x, y, tx, ty, allx, ally. =========")
