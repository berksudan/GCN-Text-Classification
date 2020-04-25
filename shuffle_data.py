import random
import shutil
from typing import List, Tuple, Any
import os


def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:  # TODO: make it common
    if data_set_name not in all_data_set_names:
        raise AttributeError("Wrong data-set name, given:%r, however expected:%r" % (data_set_name, all_data_set_names))


def create_dir(dir_path: str, overwrite: bool) -> None:  # TODO: make it common
    if os.path.exists(dir_path):
        if overwrite:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        else:
            print('[WARN] directory:%r already exists, not overwritten.' % dir_path)
    else:
        os.makedirs(dir_path)


def check_paths_exists(*paths: str):  # TODO: make it common
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError('Path: {path} is not found.'.format(path=path))


def write_list_to_file(a_list: List[Any], file_path: str, file_mode: str = 'w'):  # TODO: make it common
    with open(file_path, file_mode) as f:
        f.writelines("%s\n" % item for item in a_list)


def load_corpus_info(corpus_info_path: str) -> Tuple[List[str], List[str], List[str]]:
    all_doc_info_list = [line.strip() for line in open(corpus_info_path, 'r').readlines()]
    train_doc_info_list = [doc_info for doc_info in all_doc_info_list if doc_info.split('\t')[1].endswith('train')]
    test_doc_info_list = [doc_info for doc_info in all_doc_info_list if doc_info.split('\t')[1].endswith('test')]
    return all_doc_info_list, train_doc_info_list, test_doc_info_list


def shuffle():
    # User Parameters
    data_set_name = '20ng'
    data_set_extension = 'txt'

    # Derived Parameters
    ds = data_set_name + '.' + data_set_extension
    ds_corpus = CORPUS_DIR + ds
    ds_corpus_info = CORPUS_INFO_DIR + ds
    ds_corpus_cleaned = CORPUS_CLEANED_DIR + ds

    ds_corpus_shuffled = CORPUS_SHUFFLED_DIR + ds
    ds_corpus_shuffled_train_idx = CORPUS_SHUFFLED_INDEX_DIR + data_set_name + '.train'
    ds_corpus_shuffled_test_idx = CORPUS_SHUFFLED_INDEX_DIR + data_set_name + '.test'
    ds_corpus_shuffled_info = CORPUS_SHUFFLED_INFO_DIR + ds

    # Checkers
    check_data_set(data_set_name=data_set_name, all_data_set_names=DATA_SETS)
    check_paths_exists(ds_corpus, ds_corpus_info, ds_corpus_cleaned)

    # Create dirs if not exist
    create_dir(CORPUS_SHUFFLED_DIR, overwrite=False)
    create_dir(CORPUS_SHUFFLED_INFO_DIR, overwrite=False)
    create_dir(CORPUS_SHUFFLED_INDEX_DIR, overwrite=False)

    all_doc_info_list, train_doc_info_list, test_doc_info_list = load_corpus_info(corpus_info_path=ds_corpus_info)
    cleaned_doc_lines = [line.strip() for line in open(ds_corpus_cleaned, 'r')]

    # Shuffle train ids and write to file
    train_doc_info_ids = [all_doc_info_list.index(train_doc_info) for train_doc_info in train_doc_info_list]
    random.shuffle(train_doc_info_ids)
    write_list_to_file(a_list=train_doc_info_ids, file_path=ds_corpus_shuffled_train_idx, file_mode='w')

    # Shuffle test ids and write to file
    test_doc_info_ids = [all_doc_info_list.index(test_doc_info) for test_doc_info in test_doc_info_list]
    random.shuffle(test_doc_info_ids)
    write_list_to_file(a_list=test_doc_info_ids, file_path=ds_corpus_shuffled_test_idx, file_mode='w')

    all_doc_info_ids = train_doc_info_ids + test_doc_info_ids
    # Write shuffled info to file
    shuffled_doc_info_list = [all_doc_info_list[all_doc_info_id] for all_doc_info_id in all_doc_info_ids]
    write_list_to_file(a_list=shuffled_doc_info_list, file_path=ds_corpus_shuffled_info, file_mode='w')

    # Write shuffled document files to file
    shuffled_doc_lines = [cleaned_doc_lines[all_doc_info_id] for all_doc_info_id in all_doc_info_ids]
    write_list_to_file(a_list=shuffled_doc_lines, file_path=ds_corpus_shuffled, file_mode='w')


if __name__ == '__main__':
    # Pre-Defined Parameters
    DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    CORPUS_DIR = 'data/corpus/'
    CORPUS_INFO_DIR = 'data/corpus.info/'
    CORPUS_CLEANED_DIR = 'data/corpus.cleaned/'
    CORPUS_SHUFFLED_DIR = 'data/corpus.shuffled/'
    CORPUS_SHUFFLED_INDEX_DIR = 'data/corpus.shuffled.index/'
    CORPUS_SHUFFLED_INFO_DIR = 'data/corpus.shuffled.info/'

    shuffle()

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
