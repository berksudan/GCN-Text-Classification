import pickle
from collections import OrderedDict
from shutil import rmtree
from typing import List, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from common import check_data_set
from preprocessors.configs import PreProcessingConfigs
from utils.file_ops import write_iterable_to_file, create_dir, check_paths


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


def prepare_words(ds_name: str, cfg: PreProcessingConfigs):
    ds_corpus = cfg.corpus_shuffled_dir + ds_name + cfg.data_set_extension

    # Checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(ds_corpus)

    # Create output directories
    create_dir(dir_path=cfg.corpus_shuffled_vocab_dir, overwrite=False)
    create_dir(dir_path=cfg.corpus_shuffled_word_vectors_dir, overwrite=False)

    ds_corpus_vocabulary = cfg.corpus_shuffled_vocab_dir + ds_name + '.vocab'
    ds_corpus_word_vectors = cfg.corpus_shuffled_word_vectors_dir + ds_name + '.word_vectors'
    # ###################################################

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

    print("[INFO] Vocabulary Dir='{}'".format(cfg.corpus_shuffled_vocab_dir))
    print("[INFO] Word-Vector Dir='{}'".format(cfg.corpus_shuffled_word_vectors_dir))
    print("[INFO] ========= PREPARED WORDS: Vocabulary & word-vectors extracted. =========")
