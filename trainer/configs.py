from os import getcwd
from os.path import isabs


def make_path_absolute(a_path: str) -> str:
    if isabs(a_path):
        print('[WARN] Path:{} is already absolute.'.format(a_path))
        return a_path
    current_working_dir = getcwd()
    return current_working_dir + '/' + a_path


class TrainingConfigs:
    def __init__(self):
        self.data_sets = None  # List of Valid Data-sets
        self.corpus_split_index_dir = None  # Train and Test Index of Corpus
        self.corpus_node_features_dir = None  # Node Features (x,y,tx,ty,allx) of Corpus
        self.corpus_adjacency_dir = None  # Adjacency Matrix (adj) of Corpus
        self.corpus_vocab_dir = None  # Vocabulary of Corpus

        self.model = None  # Model-Type. Options: 'gcn', 'gcn_cheby', 'dense'
        self.learning_rate = None  # Initial learning rate, e.g. 0.02
        self.epochs = None  # Number of epochs to train, e.g. 200
        self.hidden1 = None  # Number of units in hidden layer 1, e.g. 200
        self.dropout = None  # Dropout rate (1 - keep probability), e.g. 0.5
        self.weight_decay = None  # Weight for L2 loss on embedding matrix, e.g. 0.0
        self.early_stopping = None  # Tolerance for early stopping (# of epochs), e.g. 10
        self.chebyshev_max_degree = None  # Maximum Chebyshev polynomial degree, e.g. 3

    def build(self) -> 'TrainingConfigs':
        self.corpus_split_index_dir = make_path_absolute(self.corpus_split_index_dir)
        self.corpus_node_features_dir = make_path_absolute(self.corpus_node_features_dir)
        self.corpus_adjacency_dir = make_path_absolute(self.corpus_adjacency_dir)
        self.corpus_vocab_dir = make_path_absolute(self.corpus_vocab_dir)
        return self
