from os import getcwd
from os.path import isabs
from typing import List


def make_path_absolute(a_path: str) -> str:
    if isabs(a_path):
        print('[WARN] Path:{} is already absolute.'.format(a_path))
        return a_path
    current_working_dir = getcwd()
    return current_working_dir + '/' + a_path


class PreProcessingConfigs:
    DATA_SETS = None  # type: List[str]
    DATA_SET_EXTENSION = None  # type: str
    CORPUS_DIR = None
    CORPUS_CLEANED_DIR = None
    CORPUS_META_DIR = None
    CORPUS_SHUFFLED_DIR = None

    CORPUS_SHUFFLED_SPLIT_INDEX_DIR = None
    CORPUS_SHUFFLED_META_DIR = None

    CORPUS_SHUFFLED_VOCAB_DIR = None
    CORPUS_SHUFFLED_WORD_VECTORS_DIR = None
    CORPUS_SHUFFLED_NODE_FEATURES_DIR = None

    CORPUS_SHUFFLED_ADJACENCY_DIR = None

    def build(self) -> 'PreProcessingConfigs':
        self.CORPUS_DIR = make_path_absolute(self.CORPUS_DIR)
        self.CORPUS_CLEANED_DIR = make_path_absolute(self.CORPUS_CLEANED_DIR)
        self.CORPUS_META_DIR = make_path_absolute(self.CORPUS_META_DIR)
        self.CORPUS_SHUFFLED_DIR = make_path_absolute(self.CORPUS_SHUFFLED_DIR)
        self.CORPUS_SHUFFLED_SPLIT_INDEX_DIR = make_path_absolute(self.CORPUS_SHUFFLED_SPLIT_INDEX_DIR)
        self.CORPUS_SHUFFLED_META_DIR = make_path_absolute(self.CORPUS_SHUFFLED_META_DIR)
        self.CORPUS_SHUFFLED_VOCAB_DIR = make_path_absolute(self.CORPUS_SHUFFLED_VOCAB_DIR)
        self.CORPUS_SHUFFLED_WORD_VECTORS_DIR = make_path_absolute(self.CORPUS_SHUFFLED_WORD_VECTORS_DIR)
        self.CORPUS_SHUFFLED_NODE_FEATURES_DIR = make_path_absolute(self.CORPUS_SHUFFLED_NODE_FEATURES_DIR)
        self.CORPUS_SHUFFLED_ADJACENCY_DIR = make_path_absolute(self.CORPUS_SHUFFLED_ADJACENCY_DIR)
        return self
