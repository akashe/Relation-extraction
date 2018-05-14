import numpy as np
from bidirectionalLSTM_multiwindow import network_settings
import os
import pickle
import gzip
import operator

## Add all these paths in flags
## From now instead of using network settings use config

class Dataset(network_settings):
    def __init__(self,type,reduce_class):
        self.embeddings = self.get_word_embeddings()
        if reduce_class:
            self.reduce_data()  # TODO: Better ways for reducing no of classes without manual intervention
        self.left,self.mid,self.right,self.left_seqlen,self.mid_seqlen,self.right_seqlen,self.y = [],[],[],[],[],[],[]
        self.preen(type)


    def get_batch(self):
        ## implement using generator but wait I am sending through list
        ## so I already have the full data
        ## Do something for reading directly from file
        for i in range(len(self.x)/self.batch_size):
            yield self.left[i*self.batch_size:(i+1)*self.batch_size],\
            self.mid[i*self.batch_size:(i+1)*self.batch_size],\
            self.right[i*self.batch_size:(i+1)*self.batch_size],\
            self.left_seqlen[i*self.batch_size:(i+1)*self.batch_size],\
            self.mid_seqlen[i*self.batch_size:(i+1)*self.batch_size],\
            self.right_seqlen[i*self.batch_size:(i+1)*self.batch_size],\
            self.y[i*self.batch_size:(i+1)*self.batch_size]

    def get_pos(self,sentence,e1,e2):
        return sentence.index(e1),sentence.index(e2)


    def preen(self,type):
        relations = pickle.load(open("../data/relations", "rb")) ## Oh god this is a such a bad flow
        num_of_classes = len(relations)
        with gzip.open("data/"+type+".txt.gz", "rb") as f:
            for line in f:
                line = line.decode("ascii").strip().split("\t")
                relation = line[4]
                sentence = line[5]
                e1 = line[2]
                e2 = line[3]
                if relation in relations:
                    try:
                        if sentence.find(e2) == -1 or sentence.find(e1) == -1:
                            raise Exception
                        if sentence.find(e2) > sentence.find(e1):
                            temp = e1
                            e1 = e2
                            e2 = temp
                        e1pos,e2pos = self.get_pos(sentence.strip(),e1,e2)
                    except Exception:
                        continue

                    left = [self.embeddings(word) if word in self.embeddings else self.embeddings['UNK'] for word in sentence[:e1pos-1].strip().split(" ")] # Adjusting
                    mid = [self.embeddings(word) if word in self.embeddings else self.embeddings['UNK'] for word in sentence[e1pos+len(e1)-1:e2pos].strip().split(" ")]
                    right = [self.embeddings(word) if word in self.embeddings else self.embeddings['UNK'] for word in sentence[e2pos+len(e2)-1:].strip().split(" ")]

                    if len(left)>self.max_left_window:
                        continue
                    if len(mid)>self.max_mid_window:
                        continue
                    if len(right)>self.max_right_window:
                        continue

                    self.left.append(left)
                    self.right.append(right)
                    self.mid.append(mid)
                    self.left_seqlen.append(len(left))
                    self.right_seqlen.append(len(right))
                    self.mid_seqlen.append(len(mid))

                    label = [ 0 for i in range(num_of_classes)]
                    label[relations[relation]]=1
                    self.y.append(label)

    def reduce_data(self):
        data_loc = ["../data/train.txt.gz","../data/test.txt.gz"]
        relation_to_count = {}
        for q in data_loc:
            with gzip.open(q, "rb") as f:
                for line in f:
                    try:
                        line = (line.decode('ascii').strip().split("\t"))
                        relation = line[4]
                        sentence = line[5]
                        if len(sentence.split(" ")) > 50:
                            continue
                    except UnicodeDecodeError:
                        continue
                    if not relation in relation_to_count:
                        relation_to_count[relation] = 1
                    else:
                        relation_to_count[relation] += 1

        top_relations = {}
        relation_count = 0
        for k in sorted(relation_to_count.items(), key=operator.itemgetter(1), reverse=True)[0:reduced_count]:
            top_relations[k[0]] = relation_count
            relation_count += 1

        with open("../data/relations", "wb") as f:
            pickle.dump(obj=top_relations, file=f)

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