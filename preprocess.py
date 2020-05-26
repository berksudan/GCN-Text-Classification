from preprocessors.build_adjacency import build_adjacency
from preprocessors.build_node_features import build_node_features
from preprocessors.clean_data import clean_data
from preprocessors.configs import PreProcessingConfigs
from preprocessors.prepare_words import prepare_words
from preprocessors.shuffle_data import shuffle_data

# Create pre-processing configs
conf = PreProcessingConfigs()
conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'cora']
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

# Start pre-processing
clean_data(ds_name='R52', rare_count=5, cfg=conf)
shuffle_data(ds_name='R52', cfg=conf)
prepare_words(ds_name='R52', cfg=conf)
build_node_features(ds_name='R52', validation_ratio=0.10, use_predefined_word_vectors=False, cfg=conf)
build_adjacency(ds_name='R52', cfg=conf)
