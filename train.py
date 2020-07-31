from sys import argv

from trainer.configs import TrainingConfigs
from trainer.train_model import train_model


def create_training_cfg() -> TrainingConfigs:
    conf = TrainingConfigs()
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'cora', 'citeseer', 'pubmed']
    conf.corpus_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_node_features_dir = 'data/corpus.shuffled/node_features/'
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
    return conf


def train(ds: str, training_cfg: TrainingConfigs):
    # Start training
    train_model(ds_name=ds, is_featureless=True, cfg=training_cfg)


if __name__ == '__main__':
    trn_cfg = create_training_cfg()
    if len(argv) < 2:
        raise Exception("Dataset name cannot be left blank. Must be one of datasets:%r." % trn_cfg.data_sets)
    ds_name = argv[1]
    train(ds=ds_name, training_cfg=trn_cfg)
