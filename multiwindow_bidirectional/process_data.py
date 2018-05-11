import numpy as np
from bidirectionalLSTM_multiwindow import network_settings
import os
import pickle
import gzip

## Add all these paths in flags

class Dataset(network_settings):
    def __init__(self,type):
        self.embeddings = self.get_word_embeddings()

    def get_batch(self):
        pass

    def preen(self):
        pass

    def get_word_embeddings(self):
        if (os.listdir("../data").__contains__("embeddings.npy")):
            return pickle.load(open("../data/word_to_number", "rb"))
        else:
            word_to_number = {}
            embeddings = []
            dim = 50

            with gzip.open("../data/vec.txt.gz", "rb") as f:
                k = 0
                for i, line in enumerate(f):
                    if (i != 0):
                        line = line.strip().split()
                        try:
                            word = line[0].decode('ascii')
                        except UnicodeDecodeError:
                            continue
                        embeddings.append([float(j) for j in line[1:]])
                        word_to_number[word] = k
                        k += 1
                embeddings.append(np.random.normal(size=dim, loc=0, scale=0.05))
                word_to_number['UNK'] = k
                k += 1
                embeddings.append(np.random.normal(size=dim, loc=0, scale=0.05))
                word_to_number['PAD'] = k
            np.save("data/embeddings.npy", embeddings)
            pickle.dump(obj=word_to_number, file=open("data/word_to_number", "wb"))
            return word_to_number

    def __next__(self):
        pass