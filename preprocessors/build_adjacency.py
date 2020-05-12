import pickle
from collections import Counter
from math import log
from typing import List, Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from common import check_data_set
from preprocessors.preprocessing_configs import PreProcessingConfigs
from utils.file_ops import create_dir, check_paths
from utils.other_utils import flatten_nested_iterables


def extract_word_to_doc_ids(docs_of_words: List[List[str]]) -> Dict[str, List[int]]:
    """Extracted the document ids where unique words appeared."""
    word_to_doc_ids = {}
    for doc_id, words in enumerate(docs_of_words):
        appeared_words = set()
        for word in words:
            if word not in appeared_words:
                if word in word_to_doc_ids:
                    word_to_doc_ids[word].append(doc_id)
                else:
                    word_to_doc_ids[word] = [doc_id]
                appeared_words.add(word)
    return word_to_doc_ids


def extract_word_to_doc_counts(word_to_doc_ids: Dict[str, List[int]]) -> Dict[str, int]:
    return {word: len(doc_ids) for word, doc_ids in word_to_doc_ids.items()}


def extract_windows(docs_of_words: List[List[str]], window_size: int) -> List[List[str]]:
    """Word co-occurrence with context windows"""
    windows = []
    for doc_words in docs_of_words:
        doc_len = len(doc_words)
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)
    return windows


def extract_word_counts_in_windows(windows_of_words: List[List[str]]) -> Dict[str, int]:
    """Find the total count of unique words in each window, each window is bag-of-words"""
    bags_of_words = map(set, windows_of_words)
    return Counter(flatten_nested_iterables(bags_of_words))


def extract_word_ids_pair_to_counts(windows_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Dict[str, int]:
    word_ids_pair_to_counts = Counter()
    for window in windows_of_words:
        for i in range(1, len(window)):
            word_id_i = word_to_id[window[i]]
            for j in range(i):
                word_id_j = word_to_id[window[j]]
                if word_id_i != word_id_j:
                    word_ids_pair_to_counts.update(['{},{}'.format(word_id_i, word_id_j),
                                                    '{},{}'.format(word_id_j, word_id_i)])
    return dict(word_ids_pair_to_counts)


def extract_pmi_word_weights(windows_of_words: List[List[str]], word_to_id: Dict[str, int], vocab: List[str],
                             train_size: int) -> Tuple[List[int], List[int], List[float]]:
    """Calculate PMI as weights"""
    weight_rows = []  # type: List[int]
    weight_cols = []  # type: List[int]
    pmi_weights = []  # type: List[float]

    num_windows = len(windows_of_words)
    word_counts_in_windows = extract_word_counts_in_windows(windows_of_words=windows_of_words)
    word_ids_pair_to_counts = extract_word_ids_pair_to_counts(windows_of_words, word_to_id)

    for word_id_pair, count in word_ids_pair_to_counts.items():
        word_ids_in_str = word_id_pair.split(',')
        word_id_i, word_id_j = int(word_ids_in_str[0]), int(word_ids_in_str[1])
        word_i, word_j = vocab[word_id_i], vocab[word_id_j]
        word_freq_i, word_freq_j = word_counts_in_windows[word_i], word_counts_in_windows[word_j]
        pmi_score = log((1.0 * count / num_windows) / (1.0 * word_freq_i * word_freq_j / (num_windows * num_windows)))
        if pmi_score > 0.0:
            weight_rows.append(train_size + word_id_i)
            weight_cols.append(train_size + word_id_j)
            pmi_weights.append(pmi_score)
    return weight_rows, weight_cols, pmi_weights


def extract_cosine_similarity_word_weights(vocab: List[str], train_size: int,
                                           word_vec_path: str) -> Tuple[List[int], List[int], List[float]]:
    """Calculate Cosine Similarity of Word Vectors as weights"""
    word_vectors = pickle.load(file=open(word_vec_path, 'rb'))  # type: Dict[str,List[float]]

    weight_rows = []  # type: List[int]
    weight_cols = []  # type: List[int]
    cos_sim_weights = []  # type: List[float]

    for i, word_i in enumerate(vocab):
        for j, word_j in enumerate(vocab):
            if word_i in word_vectors and word_j in word_vectors:
                vector_i = np.array(word_vectors[word_i])
                vector_j = np.array(word_vectors[word_j])
                similarity = 1.0 - cosine(vector_i, vector_j)
                if similarity > 0.9:
                    print(word_i, word_j, similarity)
                    weight_rows.append(train_size + i)
                    weight_cols.append(train_size + j)
                    cos_sim_weights.append(similarity)
    return weight_rows, weight_cols, cos_sim_weights


def extract_doc_word_ids_pair_to_counts(docs_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Dict[str, int]:
    doc_word_freq = Counter()
    for doc_id, doc_words in enumerate(docs_of_words):
        for word in doc_words:
            word_id = word_to_id[word]
            doc_word_freq.update([str(doc_id) + ',' + str(word_id)])
    return dict(doc_word_freq)


def extract_tf_idf_doc_word_weights(
        adj_rows: List[int], adj_cols: List[int], adj_weights: List[float], vocab: List[str], train_size: int,
        docs_of_words: List[List[str]], word_to_id: Dict[str, int]) -> Tuple[List[int], List[int], List[float]]:
    """Extract Doc-Word weights with TF-IDF"""
    doc_word_ids_pair_to_counts = extract_doc_word_ids_pair_to_counts(docs_of_words, word_to_id)
    word_to_doc_ids = extract_word_to_doc_ids(docs_of_words=docs_of_words)
    word_to_doc_counts = extract_word_to_doc_counts(word_to_doc_ids=word_to_doc_ids)

    vocab_len = len(vocab)
    num_docs = len(docs_of_words)
    for doc_id, doc_words in enumerate(docs_of_words):
        doc_word_set = set()
        for word in doc_words:
            if word not in doc_word_set:
                word_id = word_to_id[word]
                word_ids_pair_count = doc_word_ids_pair_to_counts[str(doc_id) + ',' + str(word_id)]

                adj_rows.append(doc_id if doc_id < train_size else doc_id + vocab_len)
                adj_cols.append(train_size + word_id)

                doc_word_idf = log(1.0 * num_docs / word_to_doc_counts[vocab[word_id]])
                adj_weights.append(word_ids_pair_count * doc_word_idf)
                doc_word_set.add(word)
    return adj_rows, adj_cols, adj_weights


def build_adjacency(ds_name: str, cfg: PreProcessingConfigs):
    """Build Adjacency Matrix of Doc-Word Heterogeneous Graph"""

    # Input Files
    ds_corpus = cfg.CORPUS_SHUFFLED_DIR + ds_name + ".txt"
    ds_corpus_vocabulary = cfg.CORPUS_SHUFFLED_VOCAB_DIR + ds_name + '.vocab'
    ds_corpus_train_idx = cfg.CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.train'
    ds_corpus_test_idx = cfg.CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.test'

    # Checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.DATA_SETS)
    check_paths(ds_corpus, ds_corpus_vocabulary, ds_corpus_train_idx, ds_corpus_test_idx)

    create_dir(dir_path=cfg.CORPUS_SHUFFLED_ADJACENCY_DIR, overwrite=False)

    docs_of_words = [line.split() for line in open(file=ds_corpus)]
    vocab = open(ds_corpus_vocabulary).read().splitlines()  # Extract Vocabulary.
    word_to_id = {word: i for i, word in enumerate(vocab)}  # Word to its id.
    train_size = len(open(ds_corpus_train_idx).readlines())  # Real train-size, not adjusted.
    test_size = len(open(ds_corpus_test_idx).readlines())  # Real test-size.

    windows_of_words = extract_windows(docs_of_words=docs_of_words, window_size=20)

    # Extract word-word weights
    rows, cols, weights = extract_pmi_word_weights(windows_of_words, word_to_id, vocab, train_size)
    # As an alternative, use cosine similarity of word vectors as weights:
    #   ds_corpus_word_vectors = cfg.CORPUS_WORD_VECTORS_DIR + ds_name + '.word_vectors'
    #   rows, cols, weights = extract_cosine_similarity_word_weights(vocab, train_size, ds_corpus_word_vectors)

    # Extract word-doc weights
    rows, cols, weights = extract_tf_idf_doc_word_weights(rows, cols, weights, vocab,
                                                          train_size, docs_of_words, word_to_id)

    adjacency_len = train_size + len(vocab) + test_size
    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(adjacency_len, adjacency_len))

    # Dump Adjacency Matrix
    with open(cfg.CORPUS_SHUFFLED_ADJACENCY_DIR + "/ind.{}.adj".format(ds_name), 'wb') as f:
        pickle.dump(adjacency_matrix, f)

    print("[INFO] Adjacency Dir='{}'".format(cfg.CORPUS_SHUFFLED_ADJACENCY_DIR))
    print("[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========")
