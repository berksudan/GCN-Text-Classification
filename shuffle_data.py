import random
from typing import List, Tuple

from clean_data import check_data_set  # todo: check_dataset to commons
from utils.file_ops import create_dir, write_iterable_to_file, check_paths


def load_corpus_meta(corpus_meta_path: str) -> Tuple[List[str], List[str], List[str]]:
    all_doc_meta_list = [line.strip() for line in open(corpus_meta_path, 'r').readlines()]
    train_doc_meta_list = [doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[1].endswith('train')]
    test_doc_meta_list = [doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[1].endswith('test')]
    return all_doc_meta_list, train_doc_meta_list, test_doc_meta_list


def shuffle(ds_name: str):
    ds_corpus = CORPUS_DIR + ds_name + '.txt'
    ds_corpus_meta = CORPUS_META_DIR + ds_name + '.meta'
    ds_corpus_cleaned = CORPUS_CLEANED_DIR + ds_name + '.txt'

    ds_corpus_shuffled = CORPUS_SHUFFLED_DIR + ds_name + '.txt'
    ds_corpus_shuffled_train_idx = CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.train'
    ds_corpus_shuffled_test_idx = CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.test'
    ds_corpus_shuffled_meta = CORPUS_SHUFFLED_META_DIR + ds_name + '.meta'

    # Checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=DATA_SETS)
    check_paths(ds_corpus, ds_corpus_meta, ds_corpus_cleaned)

    # Create dirs if not exist
    create_dir(CORPUS_SHUFFLED_DIR, overwrite=False)
    create_dir(CORPUS_SHUFFLED_META_DIR, overwrite=False)
    create_dir(CORPUS_SHUFFLED_SPLIT_INDEX_DIR, overwrite=False)

    all_doc_meta_list, train_doc_meta_list, test_doc_meta_list = load_corpus_meta(corpus_meta_path=ds_corpus_meta)
    cleaned_doc_lines = [line.strip() for line in open(ds_corpus_cleaned, 'r')]

    # Shuffle train ids and write to file
    train_doc_meta_ids = [all_doc_meta_list.index(train_doc_meta) for train_doc_meta in train_doc_meta_list]
    random.shuffle(train_doc_meta_ids)
    write_iterable_to_file(an_iterable=train_doc_meta_ids, file_path=ds_corpus_shuffled_train_idx, file_mode='w')

    # Shuffle test ids and write to file
    test_doc_meta_ids = [all_doc_meta_list.index(test_doc_meta) for test_doc_meta in test_doc_meta_list]
    random.shuffle(test_doc_meta_ids)
    write_iterable_to_file(an_iterable=test_doc_meta_ids, file_path=ds_corpus_shuffled_test_idx, file_mode='w')

    all_doc_meta_ids = train_doc_meta_ids + test_doc_meta_ids
    # Write shuffled meta to file
    shuffled_doc_meta_list = [all_doc_meta_list[all_doc_meta_id] for all_doc_meta_id in all_doc_meta_ids]
    write_iterable_to_file(an_iterable=shuffled_doc_meta_list, file_path=ds_corpus_shuffled_meta, file_mode='w')

    # Write shuffled document files to file
    shuffled_doc_lines = [cleaned_doc_lines[all_doc_meta_id] for all_doc_meta_id in all_doc_meta_ids]
    write_iterable_to_file(an_iterable=shuffled_doc_lines, file_path=ds_corpus_shuffled, file_mode='w')


if __name__ == '__main__':
    # Pre-Defined Parameters
    DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    CORPUS_DIR = 'data/corpus/'
    CORPUS_META_DIR = 'data/corpus/meta/'
    CORPUS_CLEANED_DIR = 'data/corpus.cleaned/'
    CORPUS_SHUFFLED_DIR = 'data/corpus.shuffled/'
    CORPUS_SHUFFLED_SPLIT_INDEX_DIR = 'data/corpus.shuffled/split_index/'
    CORPUS_SHUFFLED_META_DIR = 'data/corpus.shuffled/meta/'

    for data_set in DATA_SETS:
        shuffle(ds_name=data_set)

# partial labeled data # fixme: what is the purpose of the code?
# train_ids = train_ids[:int(0.2 * len(train_ids))]
"""
# Read Word Vectors
from utils.utils import  loadWord2Vec

word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
"""
