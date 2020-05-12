import random
from typing import List, Tuple

from common import check_data_set
from preprocessors.preprocessing_configs import PreProcessingConfigs
from utils.file_ops import create_dir, write_iterable_to_file, check_paths


def load_corpus_meta(corpus_meta_path: str) -> Tuple[List[str], List[str], List[str]]:
    all_doc_meta_list = [line.strip() for line in open(corpus_meta_path, 'r').readlines()]
    train_doc_meta_list = [doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[1].endswith('train')]
    test_doc_meta_list = [doc_meta for doc_meta in all_doc_meta_list if doc_meta.split('\t')[1].endswith('test')]
    return all_doc_meta_list, train_doc_meta_list, test_doc_meta_list


def shuffle_data(ds_name: str, cfg: PreProcessingConfigs):
    ds_corpus = cfg.CORPUS_CLEANED_DIR + ds_name + cfg.DATA_SET_EXTENSION
    ds_corpus_meta = cfg.CORPUS_META_DIR + ds_name + '.meta'

    ds_corpus_shuffled = cfg.CORPUS_SHUFFLED_DIR + ds_name + cfg.DATA_SET_EXTENSION
    ds_corpus_shuffled_train_idx = cfg.CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.train'
    ds_corpus_shuffled_test_idx = cfg.CORPUS_SHUFFLED_SPLIT_INDEX_DIR + ds_name + '.test'
    ds_corpus_shuffled_meta = cfg.CORPUS_SHUFFLED_META_DIR + ds_name + '.meta'

    # Checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.DATA_SETS)
    check_paths(ds_corpus_meta, ds_corpus)

    # Create dirs if not exist
    create_dir(cfg.CORPUS_SHUFFLED_DIR, overwrite=False)
    create_dir(cfg.CORPUS_SHUFFLED_META_DIR, overwrite=False)
    create_dir(cfg.CORPUS_SHUFFLED_SPLIT_INDEX_DIR, overwrite=False)

    all_doc_meta_list, train_doc_meta_list, test_doc_meta_list = load_corpus_meta(corpus_meta_path=ds_corpus_meta)
    cleaned_doc_lines = [line.strip() for line in open(ds_corpus, 'r')]

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

    print("[INFO] Shuffled-Corpus Dir='{}'".format(cfg.CORPUS_SHUFFLED_DIR))
    print("[INFO] ========= SHUFFLED DATA: Corpus documents shuffled. =========")
#
# if __name__ == '__main__':
# # Pre-Defined Parameters
# # DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# # CORPUS_DIR = '../data/corpus/'
# # CORPUS_META_DIR = '../data/corpus/meta/'
# # CORPUS_CLEANED_DIR = '../data/corpus.cleaned/'
# # CORPUS_SHUFFLED_DIR = '../data/corpus.shuffled/'
# # CORPUS_SHUFFLED_SPLIT_INDEX_DIR = '../data/corpus.shuffled/split_index/'
# # CORPUS_SHUFFLED_META_DIR = '../data/corpus.shuffled/meta/'
#
# # for data_set in DATA_SETS:
# #     shuffle_data(ds_name=data_set)
