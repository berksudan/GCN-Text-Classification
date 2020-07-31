from sys import argv

from preprocessors.build_adjacency import build_adjacency
from preprocessors.build_node_features import build_node_features
from preprocessors.clean_data import clean_data
from preprocessors.configs import PreProcessingConfigs
from preprocessors.prepare_words import prepare_words
from preprocessors.shuffle_data import shuffle_data


def create_preprocessing_cfg() -> PreProcessingConfigs:
    conf = PreProcessingConfigs()
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'cora', 'citeseer', 'pubmed']
    conf.data_set_extension = '.txt'
    conf.corpus_dir = 'data/corpus/'
    conf.corpus_meta_dir = 'data/corpus/meta/'
    conf.corpus_cleaned_dir = 'data/corpus.cleaned/'
    conf.corpus_shuffled_dir = 'data/corpus.shuffled/'
    conf.corpus_shuffled_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_shuffled_meta_dir = 'data/corpus.shuffled/meta/'
    conf.corpus_shuffled_vocab_dir = 'data/corpus.shuffled/vocabulary/'
    conf.corpus_shuffled_word_vectors_dir = 'data/corpus.shuffled/word_vectors/'
    conf.corpus_shuffled_adjacency_dir = 'data/corpus.shuffled/adjacency/'
    conf.corpus_shuffled_node_features_dir = 'data/corpus.shuffled/node_features/'
    conf.build()
    return conf


def preprocess(ds: str, preprocessing_cfg: PreProcessingConfigs):  # Start pre-processing
    clean_data(ds_name=ds, rare_count=5, cfg=preprocessing_cfg)
    shuffle_data(ds_name=ds, cfg=preprocessing_cfg)
    prepare_words(ds_name=ds, cfg=preprocessing_cfg)
    build_node_features(ds_name=ds, validation_ratio=0.10, use_predefined_word_vectors=False, cfg=preprocessing_cfg)
    build_adjacency(ds_name=ds, cfg=preprocessing_cfg)


if __name__ == '__main__':
    prep_cfg = create_preprocessing_cfg()
    if len(argv) < 2:
        raise Exception("Dataset name cannot be left blank. Must be one of datasets:%r." % prep_cfg.data_sets)
    ds_name = argv[1]
    preprocess(ds=ds_name, preprocessing_cfg=prep_cfg)
