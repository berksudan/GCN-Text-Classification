import pickle as pkl
from collections import OrderedDict
from typing import List, Dict, MutableMapping

import numpy as np
import scipy.sparse as sp

"""
# Read Word Vectors
from utils.utils import  loadWord2Vec

word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
"""


# ######################## todo: breakpoint: shuffling is over ##################################33
def extract_word_to_doc_ids(docs_of_words: List[List[str]], vocabulary: List[str]) -> Dict[str, List[int]]:
    """Extracted the document ids where unique words appeared."""
    word_to_doc_ids = dict.fromkeys(vocabulary, [])  # Initiate word_to_doc_id_list
    for doc_id, words in enumerate(docs_of_words):
        words_in_doc = set(words)
        for word in words_in_doc:
            word_to_doc_ids[word].append(doc_id)
    return word_to_doc_ids


def extract_word_to_doc_counts(word_to_doc_ids: Dict[str, List[int]]) -> Dict[str, int]:
    return {word: len(doc_ids) for word, doc_ids in word_to_doc_ids.items()}


def compute_x(docs_of_words: List[List[str]], train_size: int, word_emb_dim: int,
              word_vectors: MutableMapping[str, np.ndarray]) -> sp.csr_matrix:
    """ x: feature vectors of training docs, no initial features """
    data_x = []
    for i in range(train_size):
        doc_vec = np.zeros(word_emb_dim, dtype=float)  # Initialize
        words = docs_of_words[i]
        for word in words:
            if word in word_vectors:
                doc_vec += word_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_x.extend(mean_doc_vec)

    row_indexes = np.array([[i] * word_emb_dim for i in range(train_size)]).flatten().tolist()
    col_indexes = list(range(word_emb_dim)) * train_size
    return sp.csr_matrix((data_x, (row_indexes, col_indexes)), shape=(train_size, word_emb_dim))


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


def compute_tx(docs_of_words: List[List[str]], test_size: int, real_train_size: int, word_emb_dim: int,
               word_vectors: MutableMapping[str, np.ndarray]) -> sp.csr_matrix:
    """ tx: feature vectors of test docs, no initial features """
    data_tx = []
    for i in range(test_size):
        doc_vec = np.zeros(word_emb_dim, dtype=float)  # Initialize
        words = docs_of_words[i + real_train_size]
        for word in words:
            if word in word_vectors:
                doc_vec += word_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_tx.extend(mean_doc_vec)

    row_indexes = np.array([[i] * word_emb_dim for i in range(test_size)]).flatten().tolist()
    col_indexes = list(range(word_emb_dim)) * test_size
    return sp.csr_matrix((data_tx, (row_indexes, col_indexes)), shape=(test_size, word_emb_dim))


def compute_allx(docs_of_words: List[List[str]], real_train_size: int, vocab: List[str],
                 word_vectors: MutableMapping[str, np.ndarray]) -> sp.csr_matrix:
    expanded_word_vectors = np.load("/home/iceking/Desktop/exp22.npy")  # todo: del
    print("1",len(expanded_word_vectors))
    print("2",len(word_vectors))
    word_embeddings_dim = len(iter(word_vectors.values()).__next__())
    # allx: the feature vectors of both labeled and unlabeled training instances
    # (a superset of x)
    # unlabeled training instances -> words
    data_allx = []
    row_size = real_train_size + len(vocab)

    for i in range(real_train_size):
        doc_vec = np.zeros(word_embeddings_dim, dtype=float)  # Initialize
        words = docs_of_words[i]
        for word in words:
            if word in word_vectors:
                doc_vec += word_vectors[word]

        mean_doc_vec = (doc_vec / len(words)).tolist()
        data_allx.extend(mean_doc_vec)

    for i in range(len(vocab)):
        for j in range(word_embeddings_dim):
            data_allx.append(expanded_word_vectors.item((i, j)))

    data_allx = np.array(data_allx)
    row_indexes = np.array([[i] * word_embeddings_dim for i in range(row_size)]).flatten()
    col_indexes = np.array(list(range(word_embeddings_dim)) * row_size)

    return sp.csr_matrix((data_allx, (row_indexes, col_indexes)), shape=(row_size, word_embeddings_dim))


def compute_ally(doc_meta_list: List[str], real_train_size: int, doc_labels: List[str], vocab_size: int) -> np.ndarray:
    ally = []
    for i in range(real_train_size):
        doc_meta = doc_meta_list[i]
        label = doc_meta.split('\t')[2]
        one_hot_encoded_label = [0] * len(doc_labels)
        label_index = doc_labels.index(label)
        one_hot_encoded_label[label_index] = 1
        ally.append(one_hot_encoded_label)

    zero_filled_one_hot_for_vocabulary = [[0] * len(doc_labels) for _ in range(vocab_size)]
    ally.extend(zero_filled_one_hot_for_vocabulary)
    return np.array(ally)


def load_word_to_word_vectors(path: str) -> MutableMapping[str, np.ndarray]:
    word_vectors_as_list = pkl.load(file=open(path, 'rb'))  # type: MutableMapping[str,List[float]]
    return OrderedDict((word, np.array(vec_lst)) for word, vec_lst in word_vectors_as_list.items())


def expand_w_vectors(w_vectors: MutableMapping[str, np.ndarray], vocab: List[str]) -> MutableMapping[str, np.ndarray]:
    """
    > Expand word vectors with words which are not included in word_vectors but in vocabulary
    > If a word is not in given word_vectors, initialize with random uniform distribution bw -0.01, 0.01
    """
    print("embddd")
    word_embeddings_dim = len(iter(w_vectors.values()).__next__())
    expanded_word_vectors = OrderedDict()
    for i, word in enumerate(vocab):
        if word in w_vectors:
            expanded_word_vectors[word] = w_vectors[word]
        else:
            print("ppppppppppp")
            expanded_word_vectors[word] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)
    print("=)=)",expanded_word_vectors == w_vectors)
    return expanded_word_vectors


def build_word_graphs(ds_name: str, train_ratio: float = 1.0):
    # TODO: try train-set ratio alternatives: 0.9, 0.5

    ds_corpus = CORPUS_DIR + ds_name + '.txt'
    ds_corpus_meta = CORPUS_META_DIR + ds_name + '.meta'
    ds_corpus_vocabulary = CORPUS_VOCABULARY_DIR + ds_name + '.vocab'
    ds_corpus_word_vectors = CORPUS_WORD_VECTORS_DIR + ds_name + '.word_vectors'

    ds_corpus_train_idx = CORPUS_SPLIT_INDEX_DIR + ds_name + '.train'
    ds_corpus_test_idx = CORPUS_SPLIT_INDEX_DIR + ds_name + '.test'

    # Retrieve doc meta ids  # todo: get from pickle or calculate here and use.
    train_doc_meta_ids = list(map(int, open(ds_corpus_train_idx, 'r').read().splitlines()))
    test_doc_meta_ids = list(map(int, open(ds_corpus_test_idx, 'r').read().splitlines()))

    vocabulary = open(ds_corpus_vocabulary).read().splitlines()  # Extract vocabulary
    doc_meta_list = open(file=ds_corpus_meta, mode='r').read().splitlines()  # Extract Meta List
    doc_labels = list(OrderedDict.fromkeys(doc_meta.split()[2] for doc_meta in open(ds_corpus_meta)))  # Extract labels

    # Adjust train size, for different training rates, for example: use 90% of training set
    real_train_size = len(train_doc_meta_ids)
    adjusted_train_size = len(train_doc_meta_ids) - int((1 - train_ratio) * real_train_size)
    test_size = len(test_doc_meta_ids)

    # real_train_doc_names = shuffled_doc_info_list[:real_train_size] # # Slice and write slice to file TODO: but why?
    # write_list_to_file(a_list=real_train_doc_names, file_path='data/' + dataset + '.real_train.name', file_mode='w')

    docs_of_words = [line.split() for line in open(file=ds_corpus)]

    tf_idf_word_vectors = load_word_to_word_vectors(path=ds_corpus_word_vectors)
    word_embeddings_dim = len(next(iter(tf_idf_word_vectors.values())))

    # Extract mean document word vectors and one hot labels of train-set
    x = compute_x(docs_of_words, adjusted_train_size, word_embeddings_dim, word_vectors=tf_idf_word_vectors)
    y = compute_y(doc_meta_list, train_size=adjusted_train_size, doc_labels=doc_labels)

    # Extract mean document word vectors and one hot labels of test-set
    tx = compute_tx(docs_of_words, test_size, real_train_size, word_embeddings_dim, word_vectors=tf_idf_word_vectors)
    ty = compute_ty(doc_meta_list, test_size=test_size, real_train_size=real_train_size, doc_labels=doc_labels)
    print("heyooxxx")

    # If you want to expand word vectors with missing words in word_vectors, comment it out
    expanded_word_vectors = expand_w_vectors(w_vectors=tf_idf_word_vectors,vocab=vocabulary)

    allx = compute_allx(docs_of_words, real_train_size, vocab=vocabulary, word_vectors=tf_idf_word_vectors)
    ally = compute_ally(doc_meta_list, real_train_size, doc_labels, vocab_size=len(vocabulary))

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # dump objects
    with open("/home/iceking/Desktop/data/ind.{}.x".format(ds_name), 'wb') as file:
        pkl.dump(x, file)

    with open("/home/iceking/Desktop/data/ind.{}.y".format(ds_name), 'wb') as file:
        pkl.dump(y, file)

    with open("/home/iceking/Desktop/data/ind.{}.tx".format(ds_name), 'wb') as file:
        pkl.dump(tx, file)

    with open("/home/iceking/Desktop/data/ind.{}.ty".format(ds_name), 'wb') as file:
        pkl.dump(ty, file)

    with open("/home/iceking/Desktop/data/ind.{}.allx".format(ds_name), 'wb') as file:
        pkl.dump(allx, file)

    with open("/home/iceking/Desktop/data/ind.{}.ally".format(ds_name), 'wb') as file:
        pkl.dump(ally, file)
    print("el finitoo")


# # #####################################################
# Doc word heterogeneous graph

# """
# # ds_corpus_shuffled_train_idx = CORPUS_SHUFFLED_INDEX_DIR + data_set_name + '.train'
# # ds_corpus_shuffled_test_idx = CORPUS_SHUFFLED_INDEX_DIR + data_set_name + '.test'
# # create_dir(dir_path=CORPUS_LABELS_DIR,overwrite=False)
# #
# # docs_of_words = [line.split() for line in open(ds_corpus_shuffled)]
# # word_to_doc_ids = extract_word_to_doc_ids(docs_of_words=docs_of_words, vocabulary=vocabulary)
# # word_to_doc_counts = extract_word_to_doc_counts(word_to_doc_ids=word_to_doc_ids)
# # word_to_word_idx = {word: i for i, word in enumerate(vocabulary)} # todo: maybe unnecessary, orders?
# """
# '''
#
# # word co-occurence with context windows
# window_size = 20
# windows = []
#
# for doc_words in shuffled_doc_lines:
#     words = doc_words.split()
#     length = len(words)
#     if length <= window_size:
#         windows.append(words)
#     else:
#         # print(length, length - window_size + 1)
#         for j in range(length - window_size + 1):
#             window = words[j: j + window_size]
#             windows.append(window)
#             # print(window)
#
# word_window_freq = {}
# for window in windows:
#     appeared = set()
#     for i in range(len(window)):
#         if window[i] in appeared:
#             continue
#         if window[i] in word_window_freq:
#             word_window_freq[window[i]] += 1
#         else:
#             word_window_freq[window[i]] = 1
#         appeared.add(window[i])
#
# word_pair_count = {}
# for window in windows:
#     for i in range(1, len(window)):
#         for j in range(0, i):
#             word_i = window[i]
#             word_i_id = word_id_map[word_i]
#             word_j = window[j]
#             word_j_id = word_id_map[word_j]
#             if word_i_id == word_j_id:
#                 continue
#             word_pair_str = str(word_i_id) + ',' + str(word_j_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1
#             # two orders
#             word_pair_str = str(word_j_id) + ',' + str(word_i_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1
#
# row = []
# col = []
# weight = []
#
# # pmi as weights
# num_window = len(windows)
#
# for key in word_pair_count:
#     temp = key.split(',')
#     i = int(temp[0])
#     j = int(temp[1])
#     count = word_pair_count[key]
#     word_freq_i = word_window_freq[vocabulary[i]]
#     word_freq_j = word_window_freq[vocabulary[j]]
#     pmi = log((1.0 * count / num_window) /
#               (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
#     if pmi <= 0:
#         continue
#     row.append(train_size + i)
#     col.append(train_size + j)
#     weight.append(pmi)
#
# '''
# # word vector cosine similarity as weights
# row = []
# col = []
# weight = []
# from scipy.spatial.distance import cosine
# for i in range(len(vocab)):
#     for j in range(len(vocab)):
#         if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
#             vector_i = np.array(word_vector_map[vocab[i]])
#             vector_j = np.array(word_vector_map[vocab[j]])
#             similarity = 1.0 - cosine(vector_i, vector_j)
#             if similarity > 0.9:
#                 print(vocab[i], vocab[j], similarity)
#                 row.append(train_size + i)
#                 col.append(train_size + j)
#                 weight.append(similarity)
# '''
# # doc word frequency
# doc_word_freq = {}
#
# for doc_id in range(len(shuffled_doc_lines)):
#     doc_words = shuffled_doc_lines[doc_id]
#     words = doc_words.split()
#     for word in words:
#         word_id = word_id_map[word]
#         doc_word_str = str(doc_id) + ',' + str(word_id)
#         if doc_word_str in doc_word_freq:
#             doc_word_freq[doc_word_str] += 1
#         else:
#             doc_word_freq[doc_word_str] = 1
#
# for i in range(len(shuffled_doc_lines)):
#     doc_words = shuffled_doc_lines[i]
#     words = doc_words.split()
#     doc_word_set = set()
#     for word in words:
#         if word in doc_word_set:
#             continue
#         j = word_id_map[word]
#         key = str(i) + ',' + str(j)
#         freq = doc_word_freq[key]
#         if i < train_size:
#             row.append(i)
#         else:
#             row.append(i + len(vocabulary))
#         col.append(train_size + j)
#         idf = log(1.0 * len(shuffled_doc_lines) /
#                   word_doc_freq[vocabulary[j]])
#         weight.append(freq * idf)
#         doc_word_set.add(word)
#
# node_size = train_size + len(vocabulary) + test_size
# adj = sp.csr_matrix(
#     (weight, (row, col)), shape=(node_size, node_size))
#
# dump objects

#
# with open("/home/iceking/Desktop/data/ind.{}.adj".format(dataset), 'wb') as corpus_info_file:
#     pkl.dump(adj, corpus_info_file)

if __name__ == '__main__':
    # Pre-Defined Parameters
    DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    CORPUS_DIR = 'data/corpus.shuffled/'
    CORPUS_META_DIR = 'data/corpus.shuffled/meta/'
    CORPUS_SPLIT_INDEX_DIR = 'data/corpus.shuffled/split_index/'

    CORPUS_VOCABULARY_DIR = 'data/corpus.shuffled/vocabulary/'
    CORPUS_WORD_VECTORS_DIR = 'data/corpus.shuffled/word_vectors/'

    build_word_graphs(ds_name='R8', train_ratio=0.9)
