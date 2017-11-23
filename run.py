import gzip
import numpy as np
import network
import tensorflow as tf
import operator
import pickle
import datetime

# TODO create data from word word to number
# TODO fix sentence len to max len



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
    #reduce number of classes to 15 (excluding NA also)
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
    relation_count = 0
    for k in sorted(relation_to_count.items(),key= operator.itemgetter(1),reverse=True)[1:reduced_count]:
        top_relations[k[0]]=relation_count
        relation_count+=1

    with gzip.open("data/preened_test.txt.gz","wb") as f_write:
        with gzip.open("data/test.txt.gz","rb") as f:
            for line in f:
                try:
                    relation = (line.decode('ascii').strip().split("\t"))[4]
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)

    with gzip.open("data/preened_train.txt.gz","wb") as f_write:
        with gzip.open("data/test.txt.gz","rb") as f:
            for line in f:
                try:
                    relation = (line.decode('ascii').strip().split("\t"))[4]
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)
    with open("data/relations","wb") as f:
        pickle.dump(obj=top_relations,file=f)


def get_train_data(num_of_classes):
    train_x=[]
    train_y=[]
    relations = pickle.load(open("data/relations","rb"))
    with gzip.open("data/preened_train.txt.gz","rb") as f:
        for line in f:
            line = line.decode("ascii").strip().split("\t")
            relation = line[4]
            sentence = line[5]
            if relation in relations:
                train_x.append(sentence)
                label = [ 0 for i in range(num_of_classes)]
                label[relations[relation]] = 1
                train_y.append(label)
    np.save("data/train_x.npy",train_x)
    np.save("data/train_y.npy",train_y)

def get_test_data(num_of_classes):
    test_x = []
    test_y = []
    relations = pickle.load(open("data/relations","rb"))
    with gzip.open("data/preened_test.txt.gz", "rb") as f:
        for line in f:
            line = line.decode("ascii").strip().split("\t")
            relation = line[4]
            sentence = line[5]
            if relation in relations:
                test_x.append(sentence)
                label = [0 for i in range(num_of_classes)]
                label[relations[relation]] = 1
                test_y.append(label)
    np.save("data/test_x.npy", test_x)
    np.save("data/test_y.npy", test_y)

def get_model():
    model = network.model()
    return model

def train():
    # word_embeddings = get_word_embeddings()
    # settings = network.network_settings()
    # get_train_data(settings.num_classes)
    # get_test_data(settings.num_classes)

    save_path = "./model/"

    print("Reading train and test data....")

    train_x = np.load("data/train_x.npy")
    train_y = np.load("data/train_y.npy")
    test_x = np.load("data/test_x.npy")
    test_y = np.load("data/test_y.npy")

    assert len(train_x)==len(train_y)
    assert len(test_x)==len(test_y)

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            intializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model",initializer=intializer):
                model = network.model()
            global_step = tf.Variable(0,name="global_step",trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)

            train_op = optimizer.minimize(model.total_loss,global_step=global_step)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            merged_summary = tf.merge_all_summaries()
            summary_w = tf.train.SummaryWriter("summary",sess.graph)

            batch_size = model.settings.batch_size
            test_batch_size = model.settings.test_batch_size
            num_classs = model.settings.num_classes
            epochs = model.settings.epochs

            for i in range(epochs):

                order = range(len(train_x))
                np.random.shuffle(order)

                for j in range(len(train_x)/batch_size):
                    feed_dict ={}
                    temp_x =[]
                    temp_y= []
                    for l in order[batch_size*j:min(batch_size*(j+1),len(train_x))]:
                        temp_x.append(train_x[l])
                        temp_y.append(train_y[l])

                    feed_dict[model.input_x]=temp_x
                    feed_dict[model.input_y]=temp_y

                    _,global_step,loss,accuracy,summary =sess.run([train_op,global_step,model.total_loss,model.accuracy,merged_summary],feed_dict)
                    summary_w.add_summary(accuracy,global_step)

                    if global_step % 20 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print ("{}: step {} , loss {:g}, acc {:g}".format(time_str,global_step,loss,accuracy))

                order_test = range(len(test_x))
                np.random.shuffle(order_test)
                acc =[]
                for j in range(len(test_x) / test_batch_size):
                    feed_dict = {}
                    temp_x =[]
                    temp_y =[]
                    for l in order[test_batch_size*j:min(test_batch_size*(j+1),len(test_y))]:
                        temp_x.append(test_x[l])
                        temp_y.append(test_y[l])

                    feed_dict[model.input_x] = temp_x
                    feed_dict[model.input_y] = temp_y

                    accuracy = sess.run(
                        [ model.accuracy], feed_dict)
                    acc.append(accuracy)

                print("Epoch: {}, test_Accuracy: {} ".format(i,np.mean(acc)))













train()
# get_model()
#preen_data()

