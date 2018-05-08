import gzip
import numpy as np
import ImprovedCNN_1by1
import tensorflow as tf
import operator
import pickle
import datetime
import os
import math
import random


## Idea:
## Run the CNN model

def get_word_embeddings():

    if (os.listdir("../data").__contains__("embeddings.npy")):
        return pickle.load(open("../data/word_to_number","rb"))
    else:
        word_to_number = {}
        embeddings = []
        dim = 50

        with gzip.open("data/vec.txt.gz","rb") as f:
            k=0
            for i,line in enumerate(f):
                if(i!=0):
                    line = line.strip().split()
                    try:
                        word = line[0].decode('ascii')
                    except UnicodeDecodeError:
                        continue
                    embeddings.append([float(j) for j in line[1:]])
                    word_to_number[word]= k
                    k+=1
            embeddings.append(np.random.normal(size=dim,loc=0,scale=0.05))
            word_to_number['UNK']=k
            k+=1
            embeddings.append(np.random.normal(size=dim, loc=0, scale=0.05))
            word_to_number['PAD'] = k
        np.save("../data/embeddings.npy",embeddings)
        pickle.dump(obj=word_to_number,file=open("../data/word_to_number","wb"))
        return word_to_number

def preen_data():
    #reduce number of classes to 15 (excluding NA also)
    reduced_count =  40 # [ num_classes + 1]
    relation_to_count ={}
    with gzip.open("../data/test.txt.gz","rb") as f:
        for line in f:
            try:
                line = (line.decode('ascii').strip().split("\t"))
                relation = line[4]
                sentence = line[5]
                if len(sentence.split(" "))>50:
                    continue
            except UnicodeDecodeError:
                continue
            if not relation in relation_to_count:
                relation_to_count[relation]=1
            else:
                relation_to_count[relation]+=1

    with gzip.open("../data/train.txt.gz","rb") as f:
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
                relation_to_count[relation]=1
            else:
                relation_to_count[relation]+=1

    ## Removing NA and location contains##
    relation_to_count.pop('NA')
    relation_to_count.pop('/location/location/contains')

    top_relations = {}
    relation_count = 0
    for k in sorted(relation_to_count.items(),key= operator.itemgetter(1),reverse=True)[0:reduced_count]:
        top_relations[k[0]]=relation_count
        relation_count+=1

    with gzip.open("../data/preened_test.txt.gz","wb") as f_write:
        with gzip.open("../data/test.txt.gz","rb") as f:
            for line in f:
                try:
                    line_ = (line.decode('ascii').strip().split("\t"))
                    relation = line_[4]
                    sentence = line_[5]
                    if len(sentence.split(" ")) > 50:
                        continue
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)

    with gzip.open("../data/preened_train.txt.gz","wb") as f_write:
        with gzip.open("../data/train.txt.gz","rb") as f:
            for line in f:
                try:
                    line_ = (line.decode('ascii').strip().split("\t"))
                    relation = line_[4]
                    sentence = line_[5]
                    if len(sentence.split(" ")) > 50:
                        continue
                except UnicodeDecodeError:
                    continue
                if relation in top_relations:
                    f_write.write(line)
    with open("../data/relations","wb") as f:
        pickle.dump(obj=top_relations,file=f)


def get_train_data(word_to_number):
    train_x=[]
    train_y=[]
    relations = pickle.load(open("../data/relations","rb"))
    num_of_classes =  len(relations)
    with gzip.open("../data/preened_train.txt.gz","rb") as f:
        for line in f:
            line = line.decode("ascii").strip().split("\t")
            relation = line[4]
            sentence = line[5]
            if relation in relations:
                sen = []
                for word in sentence.strip().split():
                    if word in word_to_number:
                        sen.append(word_to_number[word])
                    else:
                        sen.append(word_to_number['UNK'])
                pad_len = 50 - len(sen)
                for i in range(pad_len):
                    sen.append(word_to_number['PAD'])
                train_x.append(sen)
                label = [ 0 for i in range(num_of_classes)]
                label[relations[relation]] = 1
                train_y.append(label)
    np.save("../data/short_train_x.npy",train_x[:1000])
    np.save("../data/short_train_y.npy",train_y[:1000])
    np.save("../data/train_x.npy",train_x)
    np.save("../data/train_y.npy",train_y)

def get_test_data(word_to_number):
    test_x = []
    test_y = []
    relations = pickle.load(open("../data/relations","rb"))
    num_of_classes = len(relations)
    with gzip.open("../data/preened_test.txt.gz", "rb") as f:
        for line in f:
            line = line.decode("ascii").strip().split("\t")
            relation = line[4]
            sentence = line[5]
            if relation in relations:
                sen = []
                for word in sentence.strip().split():
                    if word in word_to_number:
                        sen.append(word_to_number[word])
                    else:
                        sen.append(word_to_number['UNK'])
                pad_len = 50 - len(sen)
                for i in range(pad_len):
                    sen.append(word_to_number['PAD'])
                test_x.append(sen)
                label = [0 for i in range(num_of_classes)]
                label[relations[relation]] = 1
                test_y.append(label)
    np.save("../data/short_test_x.npy",test_x[:200])
    np.save("../data/short_test_y.npy",test_y[:200])
    np.save("../data/test_x.npy", test_x)
    np.save("../data/test_y.npy", test_y)

def get_model():
    model = ImprovedCNN_1by1.model()
    return model

def find_arch_parameters(overfit,attempts_at_finding_parameter,lr_range):
    ## Adding efficient parameter search
    ## IDEA
    ## check initial loss without regulirization
    ## loss increases with regularization
    ## try overfitting the data : if the data doesnt overfit something is wrong
    ## iterate over range of values of learning rate, epochs

    # 1) Overfitting
    if overfit:
        overfit_acc = []
        acc_,_ = run_model("short",0.01,30,True,False)
        if not np.mean(acc_[-5:])>0.9:
            print("Data not overfitting something wrong with model")
            return False
        else:
            print("Model does overfit..setup is correct")

    # 2) Exploring hyperparameter values
    a,b = lr_range
    for i in range(attempts_at_finding_parameter):
        lr = math.pow(10,random.uniform(a,b))
        try:
            acc_,loss = run_model("short",lr,10,True,False)
            print("For lr={},after 10 epochs we get {} accuracy and {} loss".format(lr,np.mean(acc_),loss))
        except Exception:
            print("For lr={},model fails".format(lr))

    # A good idea should be to use the lr with the lowest loss, but here the
    # loss is the loss of last epoch
    # Thus instead of returning a lr value from here, better to do some manual analysis

    # return learning_rate,epochs,does_overfit


def run_model(data_type,lr,epochs,parameter_check,verbose):
    # word_to_number = get_word_embeddings()
    # settings = network.network_settings()
    # get_train_data(settings.num_classes,word_to_number)
    # get_test_data(settings.num_classes,word_to_number)

    save_path = "improved_CNN/"

    if verbose:
        print("Reading train and test data....")

    if data_type == "full":
        train_x = np.load("../data/train_x.npy")
        train_y = np.load("../data/train_y.npy")
        test_x = np.load("../data/test_x.npy")
        test_y = np.load("../data/test_y.npy")
    else:
        train_x = np.load("../data/short_train_x.npy")
        train_y = np.load("../data/short_train_y.npy")
        test_x = np.load("../data/short_test_x.npy")
        test_y = np.load("../data/short_test_y.npy")

    assert len(train_x)==len(train_y)
    assert len(test_x)==len(test_y)

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            intializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model",initializer=intializer):
                # for normal networks
                # model = network.model()
                # for capsnet
                model = ImprovedCNN_1by1.model()
            global_step = tf.Variable(0,name="global_step",trainable=False)
            optimizer = tf.train.AdamOptimizer(lr)

            train_op = optimizer.minimize(model.loss,global_step=global_step)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()

            # merged_summary = tf.merge_all_summaries()
            merged_summary = tf.summary.merge_all()
            # summary_w = tf.train.SummaryWriter("summary",sess.graph)
            summary_w = tf.summary.FileWriter("summary", sess.graph)
            batch_size = model.settings.batch_size
            test_batch_size = model.settings.test_batch_size
            num_classs = model.settings.num_classes
            epochs = epochs or model.settings.epochs

            acc_= 0

            if parameter_check:
                acc_global = []

            for i in range(epochs):

                order = [o for o in range(len(train_x))]
                np.random.shuffle(order)

                for j in range(int(len(train_x)/batch_size)):
                    feed_dict ={}
                    temp_x =[]
                    temp_y= []
                    for l in order[batch_size*j:min(batch_size*(j+1),len(train_x))]:
                        temp_x.append(train_x[l])
                        temp_y.append(train_y[l])

                    # feed_dict[model.input_x]=temp_x
                    # feed_dict[model.input_y]=temp_y

                    feed_dict[model.input_x] = temp_x
                    feed_dict[model.input_y] = temp_y

                    _ ,step,loss,accuracy,summary =sess.run([train_op,global_step,model.loss,model.accuracy,merged_summary],feed_dict)
                    summary_w.add_summary(summary,step)

                    if step % 20 == 0 and verbose:
                        time_str = datetime.datetime.now().isoformat()
                        print ("{}: step {} , loss {:g}, acc {:g}".format(time_str,step,loss,accuracy))

                    if parameter_check:
                        acc_global.append(accuracy)

                if not parameter_check:
                    order_test = [w for w in range(len(test_x))]
                    np.random.shuffle(order_test)
                    acc =[]
                    for j in range(int(len(test_x) / test_batch_size)):
                        feed_dict = {}
                        temp_x =[]
                        temp_y =[]
                        for l in order_test[test_batch_size*j:min(test_batch_size*(j+1),len(test_y))]:
                            temp_x.append(test_x[l])
                            temp_y.append(test_y[l])

                        feed_dict[model.input_x] = temp_x
                        feed_dict[model.input_y] = temp_y

                        accuracy = sess.run(
                            [ model.accuracy], feed_dict)
                        acc.append(accuracy)

                    if(np.mean(acc)>acc_):
                        acc_ = np.mean(acc)
                        print('Saving model')
                        saver.save(sess,save_path,global_step=global_step)
                    if verbose:
                        print("Epoch: {}, test_Accuracy: {} Max_accuracy: {} ".format(i,np.mean(acc),acc_))

    if parameter_check:
        return acc_global,loss

##### Check Model
# get_model()

###### Before Training
# a = get_word_embeddings()
# preen_data()
# get_train_data(a)
# get_test_data(a)

## Check Architecture
# find_arch_parameters(True,20,[-3.0,1.0])

# Run Model
run_model("full",0.002,None,False,True)
