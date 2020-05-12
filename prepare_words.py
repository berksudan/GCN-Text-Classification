import pickle
from collections import OrderedDict
from shutil import rmtree
from typing import List, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.file_ops_utils import write_iterable_to_file, create_dir, check_paths_exists


def extract_word_definitions(vocabulary: List[str]) -> List[str]:
    from nltk.corpus import wordnet
    from nltk import download
    temporary_nltk_folder = 'venv/nltk_data/'
    download(info_or_id='wordnet', download_dir=temporary_nltk_folder)

    merged_definitions_of_words = []
    for word in vocabulary:
        syn_sets_of_word = wordnet.synsets(word.strip())
        word_definitions = [syn_set.definition() for syn_set in syn_sets_of_word]
        if not word_definitions:  # If list is empty, fill with '<PAD>'
            merged_definitions_of_word = '<PAD>'
        else:
            merged_definitions_of_word = ' '.join(word_definitions)
        merged_definitions_of_words.append(merged_definitions_of_word)
    rmtree(temporary_nltk_folder)
    return merged_definitions_of_words


def extract_tf_idf_word_vectors(word_definitions: List[str], max_features: int) -> List[np.ndarray]:
    tf_idf_vectorizer = TfidfVectorizer(max_features=max_features)
    tf_idf_vector_arrays = tf_idf_vectorizer.fit_transform(word_definitions).toarray()
    return tf_idf_vector_arrays


def extract_vocabulary(docs_of_words: Iterable[List[str]]) -> List[str]:
    vocabulary = OrderedDict()
    for words in docs_of_words:
        vocabulary.update((word, None) for word in words)
    return list(vocabulary.keys())


def prepare_words(ds_name: str):
    ds_corpus = CORPUS_DIR + ds_name + '.txt'

    check_paths_exists(ds_corpus)

    # Create Output directories
    create_dir(dir_path=CORPUS_VOCABULARY_DIR, overwrite=False)
    create_dir(dir_path=CORPUS_WORD_VECTORS_DIR, overwrite=False)

    ds_corpus_vocabulary = CORPUS_VOCABULARY_DIR + ds_name + '.vocab'
    ds_corpus_word_vectors = CORPUS_WORD_VECTORS_DIR + ds_name + '.word_vectors'
    # ###################################################3

    # Build vocabulary
    docs_of_words_generator = (line.split() for line in open(ds_corpus))
    vocabulary = extract_vocabulary(docs_of_words=docs_of_words_generator)
    write_iterable_to_file(an_iterable=vocabulary, file_path=ds_corpus_vocabulary, file_mode='w')

    # Extract word definitions
    word_definitions = extract_word_definitions(vocabulary=vocabulary)
    # write_iterable_to_file(word_definitions, file_path='/<>' + ds, file_mode='w+')

    # Extract & Dump word vectors
    word_vectors = extract_tf_idf_word_vectors(word_definitions=word_definitions, max_features=1000)
    word_to_word_vectors_dict = OrderedDict((word, vec.tolist()) for word, vec in zip(vocabulary, word_vectors))
    pickle.dump(obj=word_to_word_vectors_dict, file=open(ds_corpus_word_vectors, mode='wb'))

    # word_embeddings_dim = len(list(word_vectors.values())[0])  # todo: should be 1000 ?


if __name__ == '__main__':
    # Pre-Defined Parameters
    DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    CORPUS_DIR = 'data/corpus.shuffled/'

    CORPUS_VOCABULARY_DIR = 'data/corpus.shuffled/vocabulary/'
    CORPUS_WORD_VECTORS_DIR = 'data/corpus.shuffled/word_vectors/'

    for data_set_name in DATA_SETS:
        prepare_words(ds_name=data_set_name)
