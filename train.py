from trainer.configs import TrainingConfigs
from trainer.train_model import train_model

# Create training configs
conf = TrainingConfigs()
conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'cora']
conf.corpus_split_index_dir = 'data/corpus.shuffled/split_index/'
conf.corpus_node_features_dir ='data/corpus.shuffled/node_features/'
conf.corpus_adjacency_dir = 'data/corpus.shuffled/adjacency/'
conf.corpus_vocab_dir = 'data/corpus.shuffled/vocabulary/'
conf.model = 'gcn'
conf.learning_rate = 0.02
conf.epochs = 200
conf.hidden1 = 200
conf.dropout = 0.5
conf.weight_decay = 0.
conf.early_stopping = 10
conf.chebyshev_max_degree = 3
conf.build()

# Start training
train_model(ds_name='R52',is_featureless=True,cfg=conf)