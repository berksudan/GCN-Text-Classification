import time

import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics

from common import check_data_set
from trainer.configs import TrainingConfigs
from trainer.load_corpus import load_corpus
from trainer.prepare_matrices import prepare_matrices
from utils.logger import print_log


def set_seeds(seed: int, set_seed_randomly: bool = False):
    """Set seeds for reproducibility."""
    if set_seed_randomly:
        seed = np.random.randint(1, 200)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cuda():
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    # Settings
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pass  # Pass for now


def evaluate_model(model, criterion, features, labels, mask):
    t_test = time.time()
    # feed_dict_val = construct_feed_dict(
    #     features, support, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    model.eval()
    with torch.no_grad():
        logits = model(features)
        t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)


def train_model(ds_name: str, is_featureless: bool, cfg: TrainingConfigs):
    configure_cuda()
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    set_seeds(seed=2019)

    # Load corpus & unpack values
    corpus_values = load_corpus(ds_name, cfg.corpus_split_index_dir, cfg.corpus_node_features_dir,
                                cfg.corpus_adjacency_dir)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = corpus_values

    if is_featureless:
        features = sp.identity(features.shape[0])

    features, support, num_supports, model_func = prepare_matrices(features, adj, cfg.model, cfg.chebyshev_max_degree)

    # Define placeholders
    t_features = torch.from_numpy(features)
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])

    t_support = []
    for i in range(len(support)):
        # noinspection PyArgumentList
        t_support.append(torch.Tensor(support[i]))

    # if torch.cuda.is_available():
    #     model_func = model_func.cuda()
    #     t_features = t_features.cuda()
    #     t_y_train = t_y_train.cuda()
    #     t_y_val = t_y_val.cuda()
    #     t_y_test = t_y_test.cuda()
    #     t_train_mask = t_train_mask.cuda()
    #     tm_train_mask = tm_train_mask.cuda()
    #     for i in range(len(support)):
    #         t_support = [t.cuda() for t in t_support if True]

    model = model_func(input_dim=features.shape[0], support=t_support, num_classes=y_train.shape[1])

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    val_losses = []

    # Train model
    for epoch in range(cfg.epochs):
        epoch_start_time = time.time()

        # Forward pass
        logits = model(t_features)
        loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
        acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[
            1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        val_loss, val_acc, pred, labels, duration = evaluate_model(model, criterion, t_features, t_y_val, val_mask)
        val_losses.append(val_loss)

        print_log("Epoch:{:.0f}, train_loss={:.5f}, train_acc={:.5f}, val_loss={:.5f}, val_acc={:.5f}, time={:.5f}"
                  .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - epoch_start_time))

        if epoch > cfg.early_stopping and val_losses[-1] > np.mean(val_losses[-(cfg.early_stopping + 1):-1]):
            print_log("Early stopping...")
            break

    print_log("Optimization Finished!")

    # Testing
    test_loss, test_acc, pred, labels, test_duration = evaluate_model(model, criterion, t_features, t_y_test, test_mask)
    print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc,
                                                                                           test_duration))

    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(np.argmax(labels[i]))

    print_log("Test Precision, Recall and F1-Score...")
    print_log(metrics.classification_report(test_labels, test_pred, digits=4))
    print_log("Macro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print_log("Micro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

    # doc and word embeddings
    tmp = model.layer1.embedding.numpy()
    word_embeddings = tmp[train_size: adj.shape[0] - test_size]
    train_doc_embeddings = tmp[:train_size]  # include val docs
    test_doc_embeddings = tmp[adj.shape[0] - test_size:]

    print_log('Embeddings:')
    print_log('\rWord_embeddings:' + str(len(word_embeddings)))
    print_log('\rTrain_doc_embeddings:' + str(len(train_doc_embeddings)))
    print_log('\rTest_doc_embeddings:' + str(len(test_doc_embeddings)))
    print_log('\rWord_embeddings:')
    print(word_embeddings)

    # Create word-vectors and written to file # todo: commented-out
    """
    with open(cfg.corpus_vocab_dir + ds_name + '.vocab', 'r') as f:
        words = f.readlines()
        
    vocab_size = len(words)
    word_vectors = []
    for i in range(vocab_size):
        word = words[i].strip()
        word_vector = word_embeddings[i]
        word_vector_str = ' '.join([str(x) for x in word_vector])
        word_vectors.append(word + ' ' + word_vector_str)

    word_embeddings_str = '\n'.join(word_vectors) 
    with open('./data/' + ds_name + '_word_vectors.txt', 'w') as f:
        f.write(word_embeddings_str)
    """

    # Create doc vectors and written to file  # todo: commented-out
    """
    doc_vectors = []
    doc_id = 0
    for i in range(train_size):
        doc_vector = train_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vec('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += tors.append1

    for i in range(test_size):
        doc_vector = test_doc_embeddings[i]
        doc_vector_str = ' '.join([str(x) for x in doc_vector])
        doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
        doc_id += 1

    doc_embeddings_str = '\n'.join(doc_vectors)
    with open('./data/' + ds_name + '_doc_vectors.txt', 'w') as f:
        f.write(doc_embeddings_str)
    """
