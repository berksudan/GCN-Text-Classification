from collections import OrderedDict
from os import makedirs
from os.path import exists
from shutil import rmtree
from typing import Any, Iterable
from typing import List, MutableMapping

import numpy as np
import pandas as pd


def create_dir(dir_path: str, overwrite: bool) -> None:
    if exists(dir_path):
        if overwrite:
            rmtree(dir_path)
            makedirs(dir_path)
        else:
            print('[WARN] directory:%r already exists, not overwritten.' % dir_path)
    else:
        makedirs(dir_path)


def print_if_debug(func):  # Decorator for debug
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if DEBUG_MODE:
            print("[INFO]", str(value))
        return value

    return wrapper


@print_if_debug
def extract_train_test_labels(df: pd.DataFrame, test_ratio: float) -> pd.DataFrame:
    test_size = int(df.shape[0] * test_ratio)
    train_size = df.shape[0] - test_size
    train_test_labels_df = pd.DataFrame(["train"] * train_size + ["test"] * test_size)
    train_test_labels_df = train_test_labels_df.sample(frac=1).reset_index(drop=True)
    return train_test_labels_df


@print_if_debug
def add_split_info_to_df(ds_df: pd.DataFrame, train_test_labels_df: pd.DataFrame) -> pd.DataFrame:
    split_added_content_df = pd.concat([train_test_labels_df, ds_df], ignore_index=True, axis=1)
    split_added_content_df = split_added_content_df.sort_values(by=0, ascending=False, ignore_index=True)
    return split_added_content_df


@print_if_debug
def create_meta(meta_name: str, df: pd.DataFrame, doc_id_col, test_train_col, label_col):
    meta_df = df[[doc_id_col, test_train_col, label_col]]
    meta_df.to_csv(meta_name, header=None, sep='\t', index=False)


def extract_word_binaries(df: pd.DataFrame, word_bin_start: int, word_bin_end: int) -> pd.DataFrame:
    word_binaries = df.loc[:, word_bin_start:word_bin_end].reset_index(drop=True)
    word_binaries = word_binaries.T.reset_index(drop=True).T
    return word_binaries


@print_if_debug
def extract_doc_ids(df: pd.DataFrame, doc_id_col: int) -> List[int]:
    doc_ids = df.loc[:, doc_id_col]
    doc_ids.to_numpy()
    return doc_ids.to_list()


def create_corpus(corpus_name: str, list_of_word_ids: List[np.ndarray]):
    with open(corpus_name, 'w+') as fp:
        for word_ids in list_of_word_ids:
            fp.write(' '.join(word_ids.astype(str)) + '\n')


def extract_doc_ids_to_word_ids_present(word_binaries: pd.DataFrame, doc_ids: List[int]) -> MutableMapping:
    doc_ids_to_word_ids_present = OrderedDict()
    for i in range(word_binaries.shape[0]):
        word_ids_present = word_binaries.columns[word_binaries.iloc[i] == 1].to_numpy()
        doc_ids_to_word_ids_present[doc_ids[i]] = word_ids_present
    return doc_ids_to_word_ids_present


def main(ds_name: str, ds_content_path: str, text_gcn_ds_dir_path: str):
    gcn_meta_path = text_gcn_ds_dir_path + ds_name + '.meta'
    gcn_corpus_path = text_gcn_ds_dir_path + ds_name + '.txt'

    create_dir(dir_path=text_gcn_ds_dir_path, overwrite=True)
    content_df = pd.read_csv(ds_content_path, header=None, delimiter="\t",low_memory=False)

    train_test_labels_df = extract_train_test_labels(df=content_df, test_ratio=0.3)
    split_added_content_df = add_split_info_to_df(content_df, train_test_labels_df)
    create_meta(meta_name=gcn_meta_path, df=split_added_content_df, doc_id_col=1, test_train_col=0, label_col=1435)

    word_binaries = extract_word_binaries(df=split_added_content_df, word_bin_start=2, word_bin_end=1434)
    doc_ids = extract_doc_ids(df=split_added_content_df, doc_id_col=1)
    doc_ids_to_word_ids_present = extract_doc_ids_to_word_ids_present(word_binaries=word_binaries, doc_ids=doc_ids)
    create_corpus(corpus_name=gcn_corpus_path, list_of_word_ids=list(doc_ids_to_word_ids_present.values()))


if __name__ == '__main__':
    DEBUG_MODE = True
    main(ds_name='cora', ds_content_path='cora/cora.content', text_gcn_ds_dir_path='cora.gcn_dataset/')
