from nltk import word_tokenize
import os
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset
import torch

# lower case, cutword, stopwords
def read_data(data_path):
    data = pd.read_table(data_path, encoding='latin-1', sep='\t', names=['text'])
    data = data['text'].str.lower()  #  change to lower case
    data = data.str.replace('\d+', '')  # ride of digits
    for index in data.index:   # cutwords
        data[index] = ' '.join(word_tokenize(data[index]))
    return data


# vectorize tweet
def getVector_v2(cutWords, word2vec_model):
    vector_list = [word2vec_model[k] for k in cutWords if k in word2vec_model]
    vector_df = pd.DataFrame(vector_list)
    cutWord_vector = vector_df.mean(axis=0).values
    return cutWord_vector


def getVector_text(tweets, word2vec_model):
    vector_list = []
    for cutWords in tweets:
        vector_list.append(getVector_v2(cutWords, word2vec_model))

    return np.array(vector_list, dtype='float32')


def load_data(batch_size):
    # read data
    train_x_path, test_x_path = '../Data/train_X.txt', '../Data/test_X.txt'
    train_y_path, test_y_path = '../Data/train_Y.txt', '../Data/test_Y.txt'
    train_x = read_data(train_x_path)
    test_x = read_data(test_x_path)
    train_y = pd.read_table(train_y_path, encoding='latin-1', sep='\t', names=['label'])['label']
    test_y = pd.read_table(test_y_path, encoding='latin-1', sep='\t', names=['label'])['label']

    # trianing CBOW model
    if os.path.exists("./word2Vec"):
        print("loading word2vec model...\n")
        model = Word2Vec.load('wordvec')  # load word2Vec
    else:
        # merge two frame as corpus or training Word2Vec
        print('Training word2vec model...\n')
        frames = [train_x, test_x]
        corpus = pd.concat(frames)
        documents = [tweet.split() for tweet in corpus]
        model = gensim.models.word2vec.Word2Vec(size=128, window=5, min_count=5, negative=3, sample=0.001, hs=1, workers=4)
        model.build_vocab(documents)
        words = model.wv.vocab.keys()
        vocab_size = len(words)
        print("Vocab size\n", vocab_size)
        model.train(documents, total_examples=len(documents), epochs=16)
        model.save("wordvec")  # save language model


    # vecterizing text
    print('Loading text vectors...\n')
    train_x = getVector_text(train_x, model)
    test_x = getVector_text(test_x, model)
    train_y, test_y = np.array(train_y, dtype='float32'), np.array(test_y, dtype='float32')

    # loading data
    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    test_dataset = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader, test_loader
