import tensorflow as tf
import numpy as np

#####
#   This version observations:
#   a) magnitudes of v_mag is same always below 0.8 for all inputs giving very less error, giving
#      no spike in values for bij.
#   b) Proper loss function needed: softmax with cross entropy gives Nan in no time
#   c) Squashing gives differnet mag depending on square or squrt
#     ###########


####
#   Things to do:
#   a) Increase number of layers ( with proper classes)
#   b) test new loss function
#   c) Hyperparameter search script
#   d) multigpu just for the sake of it
#   e) command line interface
#   f) Memory networks(explore dynamic memory network)
#   #######



## Problem with v_mag coming in the same range for all values as bij is not much different
# TODO : Make next version for multigpu, use tf.norm, tensordot + tqdm for train progress, summary

class network_settings():
    max_sen_len = 50  # divisible by 10
    conv_kernel = 20
    conv_size = 128
    embedding_size = 50
    num_classes = 15
    batch_size = 100
    epochs = 50
    test_batch_size = 100
    caps1_kernel = 24
    caps1_len = 64
    caps1_dimension = 8
    caps1_size = 8
    caps1_filter = caps1_len * caps1_dimension
    caps2_len = num_classes
    caps2_dimension = 5
    caps1_total_num = caps1_len * caps1_size * caps1_size
    routing_iterations = 3


class model():
    def __init__(self):
        self.settings = network_settings()

        self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.max_sen_len], name="x")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_classes], name='y')

        with tf.name_scope("Embedding_lookup"):  # assumption: This operation should run on seperate gpus for each model
            self.emb_matrix = tf.get_variable(initializer=(np.load("data/embeddings.npy").astype(dtype='float32')),
                                              name="embeddings")
            self.emb_lookup = tf.nn.embedding_lookup(self.emb_matrix, self.x)
            self.conv1_input = tf.expand_dims(self.emb_lookup, -1)
            tf.summary.histogram("conv1_input",self.conv1_input)
        with tf.name_scope("Conv1"):
            ## Later check all func in contrib.layers
            ## Conv2d in layers,nn ??
            ## Check conv1d,conv2d,conv3d ?
            self.conv1_op = tf.layers.conv2d(inputs=self.conv1_input, filters=self.settings.conv_size, kernel_size=self.settings.conv_kernel, padding="VALID")
            # self.conv1_op = tf.nn.bias_add(tf.layers.batch_normalization(self.conv1_op),tf.Variable(tf.constant(0.01,shape=[self.settings.conv_size],name="conv1_bias",dtype=tf.float32)))
            # self.conv1_op = tf.nn.dropout(self.conv1_op,keep_prob=0.5)
            tf.summary.histogram("conv1_op",self.conv1_op)
        with tf.name_scope("Primary_Capsules"):
            self.caps1 = tf.layers.conv2d(inputs=self.conv1_op, filters=self.settings.caps1_filter,
                                          kernel_size=self.settings.caps1_kernel, padding="VALID")
            # self.caps1 = tf.nn.dropout(self.caps1, keep_prob=0.9)
            # self.caps1 = tf.nn.relu(tf.nn.bias_add(tf.layers.batch_normalization(self.caps1),tf.Variable(tf.constant(0.01,shape=[self.settings.caps1_filter],name="caps1_bias"))))
            ## Removing values below zero

            self.caps1 = tf.reshape(self.caps1, [-1, self.settings.caps1_total_num, self.settings.caps1_dimension])

            self.caps1 = self.squash(self.caps1, -1)
            self.caps1 = tf.expand_dims(self.caps1, axis=-2)
            tf.summary.histogram("primary_Capsules",self.caps1)
        with tf.name_scope("bij"):
            self.bij = tf.zeros([1, self.settings.caps1_total_num, self.settings.caps2_len, 1], dtype=tf.float32,
                               name="bij")

        with tf.name_scope("WIJ"):
            self.wij = tf.truncated_normal(
                [1, self.settings.caps1_total_num, self.settings.caps2_dimension, self.settings.caps1_dimension],
                stddev=0.01, dtype=tf.float32, name="WIJ", mean=0) ## magic happened once reducing scale to 0.01
            self.wij = tf.tile(self.wij, [self.settings.batch_size, 1, 1,
                                          1])  ## Avoid these operations to have different test and train batch sizes
            tf.summary.histogram("wij",self.wij)
        self.uij = tf.matmul(self.wij, self.caps1, transpose_b=True)
        tf.summary.histogram("uij",self.uij)
        for i in range(self.settings.routing_iterations):
            self.routing()

        tf.summary.histogram("bij", self.bij)
        tf.summary.histogram("sj", self.s_j)
        ## Using magnitudes to find loss
        ## Possiblilty of other normalization techniques??
        self.v_j = tf.squeeze(self.v_j, axis=1)
        self.v_mag = tf.sqrt(tf.reduce_sum(tf.square(self.v_j), axis=1))  # TODO may be try logesum
        tf.summary.histogram("v_mag",self.v_mag)
        self.y = tf.cast(self.y, tf.float32)
        with tf.name_scope("loss"):
            mplus = 0.9
            mminus = 0.1
            lambda_ = 0.5
            # correct_prediction = tf.square(tf.maximum(0., mplus - self.v_mag), name="correct_prediction")
            # absent_prediction = tf.square(tf.maximum(0. , self.v_mag - mminus), name="absent_prediction")
            #
            # self.loss = tf.add(self.y*correct_prediction,lambda_*(1.0 - self.y)*absent_prediction,name="loss")

            # correct_prediction = tf.square(tf.maximum(0. , 1 - self.v_mag))
            # wrong_prediction = tf.square(self.v_mag )
            # self.loss = tf.reduce_mean(tf.reduce_sum(tf.add(self.y*correct_prediction,(1.0 - self.y)*wrong_prediction),axis=1)) #### Recheck this part

            correct_prediction = tf.square(tf.maximum(0., mplus - self.v_mag), name="correct_prediction")
            absent_prediction = tf.square(tf.maximum(0. , self.v_mag - mminus), name="absent_prediction")

            self.loss = tf.reduce_mean(tf.reduce_sum(tf.add(self.y*(1-correct_prediction),lambda_*(1.0 - self.y)*(absent_prediction),name="loss"),axis=1))
        # self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.v_mag)
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.001),
                                                              weights_list=tf.trainable_variables())
        tf.summary.histogram("model_loss",self.loss)
        self.total_loss = tf.reduce_mean(self.loss) + self.l2_loss
        tf.summary.scalar(name="total_loss", tensor=self.total_loss)

        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.v_mag, 1), tf.argmax(self.y, 1)), "float"),
                name="accuracy")
            tf.summary.scalar(name="accuracy", tensor=self.accuracy)

        print("Model_Ready")

    def routing(self):
        with tf.name_scope('forward_pass'):
            self.cij = tf.nn.softmax(self.bij, dim=3, name="cij")
            self.cij = tf.squeeze(self.cij, -1)
            self.cij = tf.expand_dims(self.cij, -2)
            self.s_j = tf.reduce_sum(tf.multiply(self.uij, self.cij), axis=1, keep_dims=True)
            self.v_j = self.squash(self.s_j, -2)  ## Check if this is correct axis
            self.v_j_ = tf.tile(self.v_j, [1, self.settings.caps1_total_num, 1, 1])
            self.bij += tf.matmul(self.v_j_, self.uij, transpose_a=True) ## This is always zero??

    def squash(self, s, axis):
        sum = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(s), axis=axis, keep_dims=True))
        # sqrt = tf.sqrt(sum)
        n_ = tf.square(s) / (1. + tf.square(s))  ## This is reducing the value
        u_ = s / sum
        return n_ * u_
        # sum = tf.reduce_sum(input_tensor=tf.square(s), axis=axis, keep_dims=True)
        # sqrt = tf.sqrt(sum)
        # # n_ = tf.square(sum) / (1. + tf.square(sum)) ## This is reducing the value
        # n_ = sum / (1. + sum)  ## This is reducing the value
        # u_ = s / sqrt
        # return n_ * u_


# m = model()