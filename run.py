import gzip
import numpy as np
import network
import data_extractor
import tensorflow as tf
import operator

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
    #reduce number of classes to 16
    reduced_count = 16
    relation_to_count ={}
    with gzip.open("data/test.txt.gz","rb") as f:
        for line in f:
            try:
                relation = (line.decode('ascii').strip().split("\t"))[4]
            except UnicodeDecodeError:
                continue
            if not relation in relation_to_count:
                relation_to_count[relation]=1
            else:
                relation_to_count[relation]+=1

    with gzip.open("data/train.txt.gz","rb") as f:
        for line in f:
            try:
                relation = (line.decode('ascii').strip().split("\t"))[4]
            except UnicodeDecodeError:
                continue
            if not relation in relation_to_count:
                relation_to_count[relation]=1
            else:
                relation_to_count[relation]+=1

    top_relations = {}
    for k in sorted(relation_to_count.items(),key= operator.itemgetter(1),reverse=True)[:reduced_count]:
        top_relations[k[0]]=1
        
    with gzip.open("data/preenedtest.txt.gz","wb") as f_write:
        with gzip.open("data/test.txt.gz","rb") as f:
            for line in f:
                try:
                    relation = (line.decode('ascii').strip().split("\t"))[4]
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)

    with gzip.open("data/preenedtrain.txt.gz","wb") as f_write:
        with gzip.open("data/test.txt.gz","rb") as f:
            for line in f:
                try:
                    relation = (line.decode('ascii').strip().split("\t"))[4]
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)

def get_train_data():
    pass

def get_test_data():
    pass

def get_model():
    model = network.model()
    return model

def train():
    word_embeddings = get_word_embeddings()



# train()
# get_model()
preen_data()
