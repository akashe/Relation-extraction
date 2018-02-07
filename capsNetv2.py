import tensorflow as tf
import numpy as np

#####
#   Improve last model
#   a) Dimensions of u and v needn't be same for dot product
#   b) Using seperate W for each cell in v
#   c) Using tf.split instead of slicing operations
#   d) Use newer tf versions
#   e) Design class structure to accomodate multiple layers - TODO
#   f) avoid multiple reshape operations for matmul and tile for mul
#   g) Not keeping multiple dimensions between. - Dimensions shouls represent some quantities
#   h) squashed outputs from primary layer
#   i) seperate squashing for each secondary layer capsule
#   j) tf loops - TODO
#   k) tf.matmul((..16,1),(..16,1))
#   l) check norm and slice operations
#  ########

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
    caps1_kernel = 16
    caps1_len = 64
    caps1_dimension = 8
    caps1_filter = caps1_len*caps1_dimension
    caps2_len = num_classes
    caps2_dimension = 5
    caps1_total_num = caps1_len*caps1_kernel*caps1_kernel
    routing_iterations = 3

class model():
    def __init__(self):
        self.settings = network_settings()

        self.x = tf.placeholder(dtype=tf.int32,shape=[None,self.settings.max_sen_len],name="x")
        self.y = tf.placeholder(dtype=tf.int32,shape=[None,self.settings.num_classes],name='y')

        with tf.name_scope("Embedding_lookup"): # assumption: This operation should run on seperate gpus for each model
            self.emb_matrix = tf.get_variable(initializer=(np.load("data/embeddings.npy").astype(dtype='float32')),name="embeddings")
            self.emb_lookup = tf.nn.embedding_lookup(self.emb_matrix,self.x)
            self.conv1_input = tf.expand_dims(self.emb_lookup,-1)

        with tf.name_scope("Conv1"):
            ## Later check all func in contrib.layers
            ## Conv2d in layers,nn ??
            ## Check conv1d,conv2d,conv3d ?
            self.conv1_op = tf.layers.conv2d(inputs=self.conv1_input,filters=self.settings.conv_size,kernel_size=
                                             self.settings.conv_kernel,padding="VALID",activation=tf.nn.relu)

        with tf.name_scope("Primary_Capsules"):
            self.caps1 = tf.layers.conv2d(inputs=self.conv1_op,filters=self.settings.caps1_filter,kernel_size=self.settings.caps1_kernel
                                          ,padding="VALID",activation=tf.nn.relu)
            ## Removing values below zero
            self.caps1 = tf.reshape(self.caps1,[-1,self.settings.caps1_total_num,self.settings.caps1_dimension])

            self.caps1 = self.squash(self.caps1,-1)
            self.caps1 = tf.expand_dims(self.caps1, axis=-2)

        with tf.name_scope("bij"):
            self.bij = tf.zeros([1,self.settings.caps1_total_num,self.settings.caps2_len,1],dtype=tf.float32,name="bij")

        with tf.name_scope("WIJ"):
            self.wij = tf.truncated_normal(
                [1, self.settings.caps1_total_num, self.settings.caps2_dimension, self.settings.caps1_dimension],
                stddev=1.0, dtype=tf.float32, name="WIJ")
            self.wij = tf.tile(self.wij, [self.settings.batch_size, 1, 1, 1]) ## Avoid these operations to have different test and train batch sizes

        self.uij = tf.matmul(self.wij, self.caps1,transpose_b=True)
        for i in range(self.settings.routing_iterations):
            self.routing()

        ## Using magnitudes to find loss
        ## Possiblilty of other normalization techniques??
        self.v_j= tf.squeeze(self.v_j,axis=1)
        self.v_mag = tf.reduce_sum(tf.square(self.v_j),axis=1) # TODO may be try logesum

        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.v_mag)
        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        self.total_loss = tf.reduce_mean(self.loss) + self.l2_loss
        tf.summary.scalar(name="total_loss", tensor = self.total_loss)

        with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.v_mag, 1), tf.argmax(self.y, 1)), "float"),
                name="accuracy")
            tf.summary.scalar(name="accuracy", tensor = self.accuracy)

        print("Model_Ready")

    def routing(self):
        with tf.name_scope('forward_pass'):
            self.cij = tf.nn.softmax(self.bij, axis=3, name="cij")
            self.cij = tf.squeeze(self.cij,-1)
            self.cij = tf.expand_dims(self.cij,-2)
            self.s_j = tf.reduce_sum(tf.multiply(self.uij,self.cij),axis=1,keep_dims=True)
            self.v_j = self.squash(self.s_j,-2) ## Check if this is correct axis
            self.v_j_ = tf.tile(self.v_j,[1,self.settings.caps1_total_num,1,1])
            self.bij += tf.matmul(self.v_j_,self.uij,transpose_a=True)


    def squash(self,s,axis):
        sum = tf.reduce_sum(input_tensor=tf.square(s),axis=axis,keep_dims=True)
        sqrt = tf.sqrt(sum)
        n_ = sum/(1. + sum)
        u_ = s/sqrt
        return n_*u_

# m = model()


