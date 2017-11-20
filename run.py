import gzip
import numpy as np
import network
import data_extractor
import tensorflow as tf

def get_word_embeddings():
    word_to_number = {}
    embeddings = []
    with gzip.open("data/vec.txt.gz","rb") as f:
        for i,line in enumerate(f):
            if(i!=0):
                line = line.strip().split()
                try:
                    word = line[0].decode('ascii')
                except UnicodeDecodeError:
                    continue
                embeddings.append([float(j) for j in line[1:]])
                word_to_number[word]= i
    np.save("data/embeddings.npy",embeddings)
    return word_to_number

def preen_data():
    #reduce number of classes to 40
    pass

def get_train_data():
    pass

def get_test_data():
    pass

def get_model():
    pass

def train():
    word_embeddings = get_word_embeddings()



train()

