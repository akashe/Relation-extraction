import tensorflow as tf
import numpy as np
from utils import create_if_not_there_dir
import tqdm

# Options: create more elusive code using flags,evaluator and graph handler and ? or just go with the flow??
# Do gradient clipping

## Observations:
# tf.get_Variable doesnt accept a tensor as an initial input .. WTF
# its better to give a var_Scope to prevent stupid errors while creating
# back prop variables.
# giving placeholder directly in feed dict

## Nt sure but do i have to pad till maxlen for lstm??


## Accuracies
# for 40 classes and 0.001 lr and 0.001 l2 after 100 epoch accuracy not greater than 50!!




class network_settings(object):
    # network properties
    max_left_window = 20
    max_right_window = 20
    max_mid_window = 20
    sequence_length = True
    embedding_size = 50
    share_weights = True
    hidden_units = 128

    # training properties
    num_epochs = 100
    learning_rate = 0.001
    num_classes = 40
    batch_size = 50
    eval_period = 50
    summary_period =10

class model(network_settings):
    def __init__(self):
        self.left_x = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_left_window],name='left_x')
        self.mid_x = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_mid_window],name="mid_x")
        self.right_x = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_right_window],name="right_x")

        self.left_seqlen = None
        self.mid_seqlen = None
        self.right_seqlen = None

        self.y = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.num_classes],name="y")

        embeddings = tf.get_variable(initializer=(np.load("../data/embeddings.npy").astype(dtype='float32')),
                                                   name="embeddings")

        self.left_x_e = tf.nn.embedding_lookup(embeddings,self.left_x,-1)
        self.mid_x_e = tf.nn.embedding_lookup(embeddings,self.mid_x,-1)
        self.right_x_e = tf.nn.embedding_lookup(embeddings,self.right_x,-1)

        if self.sequence_length:
            self.left_seqlen = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name="left_seqlen")
            self.mid_seqlen = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name="mid_seqlen")
            self.right_seqlen = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name="right_seqlen")

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_units,use_peepholes=True)

        ## Version 1 shares the same cell

        ## Check where to put reuse variable scope and dropout wrapper

        with tf.name_scope("bidirectional"):
            (left_fw,left_bw),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=self.left_x_e,sequence_length=self.left_seqlen,dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            (mid_fw,mid_bw),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=self.mid_x_e,sequence_length=self.mid_seqlen,dtype=tf.float32)
            (right_fw,right_bw),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,inputs=self.right_x_e,sequence_length=self.right_seqlen,dtype=tf.float32)

        with tf.name_scope("concat"):
            values = tf.concat(values=[left_fw[:,-1,:],left_bw[:,-1,:],mid_fw[:,-1,:],mid_bw[:,-1,:],right_fw[:,-1,:],right_bw[:,-1,:]],axis=-1)
            # values = tf.squeeze(values)
        #     values = tf.expand_dims()
        # with tf.name_scope("conv"):
        #     logits =
        # Not doing conv coz I just want matrix multiplication
        with tf.name_scope("wx_plus_b"):
            W = tf.Variable(initial_value=tf.truncated_normal(shape=[6*self.hidden_units,self.num_classes],mean=0.0,stddev=0.5),name="W")
            b = tf.Variable(initial_value=tf.constant(value=-1.0,dtype=tf.float32,shape=[self.num_classes]),name="b")

            logits = tf.nn.xw_plus_b(values,weights=W,biases=b)

        with  tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=logits))
            tf.summary.histogram(name="loss",values=self.loss)

            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                                                  weights_list=tf.trainable_variables())
            tf.summary.histogram(name="l2_loss",values=self.l2_loss)

            self.total_loss = self.loss+self.l2_loss

        with tf.name_scope("accuracy"):
            self.acc_ = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(self.y,1)),"float"),name="accuracy")


class Graph_Handler(model):
    def __init__(self,save_path=None,summary_path=None):
        # model saver,summary saver,global step counter,model load
        # apply gradients??
        with tf.variable_scope("model"):
            super(Graph_Handler,self).__init__()
            tf.get_variable_scope().reuse_variables()
        self.save_path = save_path+"/model.ckpt"
        self.summary_path = summary_path
        if save_path:
            create_if_not_there_dir(save_path)
        if summary_path:
            create_if_not_there_dir(summary_path)

        self.global_step = tf.Variable(0,name="global_step",trainable=False)
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.summary_path)
        self.summaries = tf.summary.merge_all()

        ## train_op
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss,global_step=self.global_step)

    def eval(self,sess,dataset):
        eval_accuracy = []  # This is very bad way of keeping track of test accuracy
        # Make better architecuture next time
        for i,data in enumerate(dataset.get_batch()):
            eval_accuracy.append(self.run(sess,data,eval=True))

        return np.mean(eval_accuracy)

    def run(self,sess,data,summary=None,eval=None,global_step = None):
        feed_dict = {}
        l, r, m, ll, rl, ml, y = data
        feed_dict[self.left_x] = l
        feed_dict[self.mid_x] = m
        feed_dict[self.right_x] = r
        feed_dict[self.left_seqlen] = ll
        feed_dict[self.right_seqlen] = rl
        feed_dict[self.mid_seqlen] = ml
        feed_dict[self.y] = y

        if summary:
            # try:
            _,accuracy,summary = sess.run([self.train_op,self.acc_,self.summaries],feed_dict=feed_dict)
            self.writer.add_summary(summary,global_step)
        else:
            _,accuracy = sess.run([self.train_op,self.acc_],feed_dict=feed_dict)

        if eval:
            accuracy = sess.run([self.acc_],feed_dict=feed_dict)

        return accuracy

    def restore(self,sess):
        latest_ckpt = tf.train.latest_checkpoint("save") # TODO remove this hardcode
        self.saver.restore(sess,latest_ckpt)

    def save(self,sess):
        self.saver.save(sess,self.save_path)

    def initialize(self,sess):
        sess.run(tf.initialize_all_variables())