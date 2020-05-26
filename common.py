from collections import Counter
from typing import Iterable, Any
from typing import List


def extract_word_counts(docs_of_words: List[List[str]]) -> Counter:
    word_counts = Counter()
    for words in docs_of_words:
        word_counts.update(words)
    return word_counts


def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:
    if data_set_name not in all_data_set_names:
        raise AttributeError("Wrong data-set name, given:%r, however expected:%r" % (data_set_name, all_data_set_names))


def flatten_nested_iterables(iterables_of_iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [item for sublist in iterables_of_iterables for item in sublist]
