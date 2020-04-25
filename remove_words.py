from collections import Counter
from typing import List, Set
from os.path import exists
from os import makedirs
from shutil import rmtree

def clean_str(a_str:str) -> str:
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:
    if data_set_name not in all_data_set_names:
        raise AttributeError("Wrong data-set name, given:%r, however expected:%r" % (data_set_name, all_data_set_names))


def create_dir(dir_path: str, overwrite: bool) -> None:
    if exists(dir_path):
        if overwrite:
            rmtree(dir_path)
            makedirs(dir_path)
        else:
            print('[WARN] directory:%r already exists, not overwritten.' % dir_path)
    else:
        makedirs(dir_path)


def retrieve_stop_words(language: str = 'english') -> Set[str]:
    from nltk.corpus import stopwords
    return set(stopwords.words(language))


def extract_word_counts(lines_of_words: List[List[str]]) -> Counter:
    w_counts = Counter()
    for words in lines_of_words:
        w_counts.update(words)
    return w_counts


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


def main(data_set: str, rare_count: int):
    # TODO #1: change '../data/corpus/' + data_set + '.txt' to more readable format
    # TODO #2: handle comment blocks at the and of the module

    with open('../data/corpus/' + data_set + '.txt', 'rb') as f:
        lines_of_words = [clean_str(line.strip().decode('latin1')).split() for line in f.readlines()]

    check_data_set(data_set_name=data_set, all_data_set_names=DATA_SETS)
    word_counts = extract_word_counts(lines_of_words=lines_of_words)
    stop_words = retrieve_stop_words(language='english')
    if data_set != 'mr':  # If data-set is 'mr', don't remove stop and rare words
        lines_of_words = remove_stop_words(lines_of_words, stop_words=stop_words)
        lines_of_words = remove_rare_words(lines_of_words, word_counts=word_counts, rare_count=rare_count)
    lines_of_words = glue_lines(lines_of_words=lines_of_words, glue_str=' ', with_strip=True)
    clean_corpus_str = '\n'.join(lines_of_words) # TODO: change with writelines version
    with open('../data/corpus/' + data_set + '.alternative.clean.txt', 'w') as f:
        f.write(clean_corpus_str)


if __name__ == '__main__':
    DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
    PARAMETERS = {
        'data_set': 'mr',
        'rare_count': 5
    }
    main(**PARAMETERS)

# Read Word Vectors
# from utils.utils import  loadWord2Vec
# word_vector_file = 'data/glove.6B/glove.6B.200d.txt'
# vocab, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])
# ######################################################
# min_len = 10000
# aver_len = 0
# max_len = 0
#
# with open('../data/corpus/' + data_set + '.clean.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         temp = line.split()
#         aver_len = aver_len + len(temp)
#         if len(temp) < min_len:
#             min_len = len(temp)
#         if len(temp) > max_len:
#             max_len = len(temp)
#
# aver_len = 1.0 * aver_len / len(lines)
# print('Min_len : ' + str(min_len))
# print('Max_len : ' + str(max_len))
# print('Average_len : ' + str(aver_len))
