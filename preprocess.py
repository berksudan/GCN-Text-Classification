from preprocessors.build_adjacency import build_adjacency
from preprocessors.build_node_features import build_node_features
from preprocessors.clean_data import clean_data
from preprocessors.prepare_words import prepare_words
from preprocessors.preprocessing_configs import PreProcessingConfigs
from preprocessors.shuffle_data import shuffle_data

prep_configs = PreProcessingConfigs()
prep_configs.DATA_SETS = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
prep_configs.DATA_SET_EXTENSION = '.txt'

prep_configs.CORPUS_DIR = 'data/corpus/'
prep_configs.CORPUS_META_DIR = 'data/corpus/meta/'

prep_configs.CORPUS_CLEANED_DIR = 'data/corpus.cleaned/'

prep_configs.CORPUS_SHUFFLED_DIR = 'data/corpus.shuffled/'
prep_configs.CORPUS_SHUFFLED_SPLIT_INDEX_DIR = 'data/corpus.shuffled/split_index/'
prep_configs.CORPUS_SHUFFLED_META_DIR = 'data/corpus.shuffled/meta/'

prep_configs.CORPUS_SHUFFLED_VOCAB_DIR = 'data/corpus.shuffled/vocabulary/'
prep_configs.CORPUS_SHUFFLED_WORD_VECTORS_DIR = 'data/corpus.shuffled/word_vectors/'
prep_configs.CORPUS_SHUFFLED_ADJACENCY_DIR = 'data/corpus.shuffled/adjacency/'

prep_configs.CORPUS_SHUFFLED_NODE_FEATURES_DIR = 'data/corpus.shuffled/node_features/'
prep_configs.build()

clean_data(ds_name='R52', rare_count=5, cfg=prep_configs)
shuffle_data(ds_name='R52', cfg=prep_configs)
prepare_words(ds_name='R52', cfg=prep_configs)
build_node_features(ds_name='R52', train_ratio=0.90, use_predefined_word_vectors=False, cfg=prep_configs)
build_adjacency(ds_name='R52', cfg=prep_configs)
