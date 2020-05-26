import re
from collections import Counter
from shutil import rmtree
from typing import List, Set

from common import extract_word_counts, check_data_set
from preprocessors.configs import PreProcessingConfigs
from utils.file_ops import create_dir, write_iterable_to_file, check_paths


def clean_str(a_str: str) -> str:
    """
    Tokenizing/string cleaning for all data-sets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", a_str)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", r" \( ", string)
    string = re.sub(r"\)", r" \) ", string)
    string = re.sub(r"\?", r" \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def retrieve_stop_words(language: str = 'english') -> Set[str]:
    temporary_nltk_folder = 'venv/nltk_data/'
    from nltk.corpus import stopwords
    from nltk import download
    download(info_or_id='stopwords', download_dir=temporary_nltk_folder)
    retrieved_stop_words = set(stopwords.words(language))
    rmtree(temporary_nltk_folder)
    return retrieved_stop_words


def remove_stop_words(lines_of_words: List[List[str]], stop_words: Set[str]) -> List[List[str]]:
    """ If a word is in stop-words, then remove it"""
    return [[word for word in line if word not in stop_words] for line in lines_of_words]


def remove_rare_words(lines_of_words: List[List[str]], word_counts: Counter, rare_count: int) -> List[List[str]]:
    """ If a word is rare, then remove it"""
    return [[word for word in line if word_counts[word] >= rare_count] for line in lines_of_words]


def glue_lines(lines_of_words: List[List[str]], glue_str: str, with_strip: bool) -> List[str]:
    if with_strip:
        return [glue_str.join(lines).strip() for lines in lines_of_words]
    else:
        return [glue_str.join(lines) for lines in lines_of_words]


def clean_data(ds_name: str, rare_count: int, cfg: PreProcessingConfigs):
    corpus_path = cfg.corpus_dir + ds_name + cfg.data_set_extension
    ds_corpus_cleaned = cfg.corpus_cleaned_dir + ds_name + cfg.data_set_extension

    # Checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(corpus_path)

    create_dir(dir_path=cfg.corpus_cleaned_dir, overwrite=False)
    docs_of_words = [clean_str(line.strip().decode('latin1')).split() for line in open(corpus_path, 'rb')]
    word_counts = extract_word_counts(docs_of_words=docs_of_words)
    stop_words = retrieve_stop_words(language='english')
    if ds_name != 'mr':  # If data-set is 'mr', don't remove stop and rare words, TODO: find why
        docs_of_words = remove_stop_words(docs_of_words, stop_words=stop_words)
        docs_of_words = remove_rare_words(docs_of_words, word_counts=word_counts, rare_count=rare_count)
    docs_of_words = glue_lines(lines_of_words=docs_of_words, glue_str=' ', with_strip=True)

    write_iterable_to_file(an_iterable=docs_of_words, file_path=ds_corpus_cleaned, file_mode='w')
    print("[INFO] Cleaned-Corpus Dir='{}'".format(cfg.corpus_cleaned_dir))
    print("[INFO] Rare-Count=<{}>".format(rare_count))
    print("[INFO] ========= CLEANED DATA: Removed rare & stop-words. =========")
