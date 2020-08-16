# Text Categorization Using GCN
Inzva AI Projects #4, Text Categorization Using GCN
![Showcase Cover with Photos](docs/pics/showcase_cover_with_photos.jpeg)

## Abstract
Our project aims to tackle text classification problem with novel approaches Graph Convolutional Networks and Graph Attention Networks using Deep Learning algorithms and Natural Language Processing Techniques.

## Build
It is applicable for only Linux distros. You can update the commands and use the equivalent ones in other distros (Mac, Windows, etc.) Executing ```buid.sh``` will create a new _virtual environment_ in the project folder and install dependencies into that. Run the following command to build: 
```bash
bash build.sh 
```
Be sure that your computer is connected to internet. It can take a while to download and install the dependendencies.

## Run
**Available Datasets:**
+ 20ng (Newsgroup Dataset)
+ R8 (Reuters News Dataset with 8 labels)
+ R52 (Reuters News Dataset with 52 labels)
+ ohsumed (Cardiovascular Diseases Abstracts Dataset)
+ mr (Movie Reviews Dataset)
+ cora (Citation Dataset)
+ citeseer (Citation Dataset)
+ pubmed (Citation Dataset)

**Preprocess:**
```bash
venv/bin/python3 preprocess.py <DATASET_NAME>
```
*Example:* ```venv/bin/python3 preprocess.py R8```

**Train:**
```bash
venv/bin/python3 train.py <DATASET_NAME>
```
*Example:* ```venv/bin/python3 train.py R8```

## Contributors
- *Berk Sudan*, [GitHub](https://github.com/berksudan), [LinkedIn](https://linkedin.com/in/berksudan/)
- *Güneykan Özgül*, [GitHub](https://github.com/guneykan/),  [LinkedIn](https://www.linkedin.com/in/guneykan-ozgul)

## Example Output
```console
$ python3 preprocess.py R8
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.cleaned/' already exists, not overwritten.
[nltk_data] Downloading package stopwords to venv/nltk_data/...
[nltk_data]   Unzipping corpora/stopwords.zip.
[INFO] Cleaned-Corpus Dir='/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.cleaned/'
[INFO] Rare-Count=<5>
[INFO] ========= CLEANED DATA: Removed rare & stop-words. =========
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/' already exists, not overwritten.
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/meta/' already exists, not overwritten.
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/split_index/' already exists, not overwritten.
[INFO] Shuffled-Corpus Dir='/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/'
[INFO] ========= SHUFFLED DATA: Corpus documents shuffled. =========
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/vocabulary/' already exists, not overwritten.
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/word_vectors/' already exists, not overwritten.
[nltk_data] Downloading package wordnet to venv/nltk_data/...
[nltk_data]   Unzipping corpora/wordnet.zip.
[INFO] Vocabulary Dir='/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/vocabulary/'
[INFO] Word-Vector Dir='/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/word_vectors/'
[INFO] ========= PREPARED WORDS: Vocabulary & word-vectors extracted. =========
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/node_features//R8' already exists, not overwritten.
[INFO] x.shape=   (4937, 300),	 y.shape=   (4937, 8)
[INFO] tx.shape=  (2189, 300),	 ty.shape=  (2189, 8)
[INFO] allx.shape=(13173, 300),	 ally.shape=(13173, 8)
[INFO] ========= EXTRACTED NODE FEATURES: x, y, tx, ty, allx, ally. =========
[WARN] directory:'/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/adjacency/' already exists, not overwritten.

[INFO] Adjacency Dir='/home/linuxuser/Desktop/GCNN - Repo/gcn_text_categorization/data/corpus.shuffled/adjacency/'
[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========
```
```console
$ python3 train.py R8
[2020/8/16 19:12:00] Epoch:1, train_loss=2.10250, train_acc=0.04436, val_loss=2.06262, val_acc=0.62044, time=4.47397
[2020/8/16 19:12:02] Epoch:2, train_loss=1.91917, train_acc=0.66235, val_loss=2.05057, val_acc=0.68978, time=2.37341
[2020/8/16 19:12:05] Epoch:3, train_loss=1.80157, train_acc=0.72514, val_loss=2.04366, val_acc=0.72263, time=2.30123
[2020/8/16 19:12:07] Epoch:4, train_loss=1.73192, train_acc=0.76727, val_loss=2.03955, val_acc=0.75365, time=2.29406

...
[2020/8/16 19:14:40] Epoch:61, train_loss=1.42440, train_acc=0.99149, val_loss=2.01033, val_acc=0.96350, time=2.95969
[2020/8/16 19:14:43] Epoch:62, train_loss=1.42407, train_acc=0.99190, val_loss=2.01033, val_acc=0.96168, time=2.79453
[2020/8/16 19:14:45] Epoch:63, train_loss=1.42377, train_acc=0.99271, val_loss=2.01034, val_acc=0.96168, time=2.56621
[2020/8/16 19:14:45] Early stopping...
[2020/8/16 19:14:45] Optimization Finished!
[2020/8/16 19:14:46] Test set results: 
[2020/8/16 19:14:46] 	 loss= 1.79863, accuracy= 0.97076, time= 0.85016
[2020/8/16 19:14:46] Test Precision, Recall and F1-Score...
[2020/8/16 19:14:47]               precision    recall  f1-score   support
[2020/8/16 19:14:47] 
[2020/8/16 19:14:47]            0     0.9826    0.9917    0.9871      1083
[2020/8/16 19:14:47]            1     0.9706    0.8148    0.8859        81
[2020/8/16 19:14:47]            2     0.9826    0.9741    0.9784       696
[2020/8/16 19:14:47]            3     0.9435    0.9669    0.9551       121
[2020/8/16 19:14:47]            4     0.8454    0.9425    0.8913        87
[2020/8/16 19:14:47]            5     0.9012    0.9733    0.9359        75
[2020/8/16 19:14:47]            6     0.9615    0.6944    0.8065        36
[2020/8/16 19:14:47]            7     1.0000    1.0000    1.0000        10
[2020/8/16 19:14:47] 
[2020/8/16 19:14:47]     accuracy                         0.9708      2189
[2020/8/16 19:14:47]    macro avg     0.9484    0.9197    0.9300      2189
[2020/8/16 19:14:47] weighted avg     0.9715    0.9708    0.9703      2189
[2020/8/16 19:14:47] 
[2020/8/16 19:14:47] Macro average Test Precision, Recall and F1-Score...
[2020/8/16 19:14:47] (0.9484369779553932, 0.9197363948390138, 0.9300186011259608, None)
[2020/8/16 19:14:47] Micro average Test Precision, Recall and F1-Score...
[2020/8/16 19:14:48] (0.9707629054362723, 0.9707629054362723, 0.9707629054362723, None)
[2020/8/16 19:14:48] Embeddings:
Word_embeddings:7688 
Train_doc_embeddings:5485
Test_doc_embeddings:2189
Word_embeddings::48] 
[[0.         0.16006473 0.         ... 0.23560655 0.03900092 0.03204489]
 [0.77342826 0.         0.         ... 0.         0.5302985  0.6264921 ]
 [0.7280124  0.2820953  0.         ... 0.         0.11174901 0.5485109 ]
 ...
 [0.3487121  0.00971795 0.16308378 ... 0.23769674 0.19000074 0.06477314]
 [0.04132688 0.10753749 0.06603149 ... 0.05606151 0.13652363 0.0443929 ]
 [0.06743763 0.23347118 0.36717737 ... 0.15584956 0.         0.        ]]
```

## References

### Papers 
+ [Liang Yao, Chengsheng Mao, Yuan Luo, 2018] Graph Convolutional Networks for Text Classification
+ [Kipf and Welling, 2017]  Semi-supervised Classification with Graph Convolutional Networks
+ [Ankit Pal, 2020] Multi-label Text Classification using Attention based Graph Neural Network
+ [Petar Velickovic, 2017] Graph Attention Networks

### Repos
+ PyTorch Implementation of Graph Attention Networks: https://github.com/Diego999/pyGAT
+ PyTorch implementation of Graph Convolutional Networks: https://github.com/tkipf/pygcn
