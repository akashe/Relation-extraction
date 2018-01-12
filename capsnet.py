import tensorflow as tf
import numpy as np


######
#   Capsule network for relation extraction
#
#    ##############


class network_settings():
    max_sen_len = 50  # divisible by 10
    conv1_filter_shape = 20
    conv1_filter_kernel = 32
    number_of_layers = 3
    embedding_size = 50
    num_classes = 15
    num_filter = 10
    batch_size = 100
    epochs = 50
    test_batch_size = 1000
    primary_capsule_size = 64
    primary_capsule_dimension =4
    primary_capsule_filter = 9
    weights_i_j = primary_capsule_dimension
    secondary_layer_dimension = 5
    secondary_layer_size = 55

class model():
    def __init__(self):
        self.settings = network_settings()
        self.primary_capsules = []
        self.primary_capsules_weights = []
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.max_sen_len], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_classes], name="input_y")

        pooled_outputs = []

        embedded_input = tf.expand_dims(
            tf.nn.embedding_lookup(tf.get_variable(initializer=(np.load("data/embeddings.npy").astype(dtype='float32')),
                                                   name="embeddings"),
                                   self.input_x), -1)

        with tf.name_scope("conv1"):
            conv1 = tf.nn.conv2d(embedded_input,tf.Variable(tf.truncated_normal([self.settings.conv1_filter_shape,self.settings.conv1_filter_shape,1,self.settings.conv1_filter_kernel],stddev=0.1,dtype=tf.float32)),strides=[1,2,2,2],padding="SAME")
            conv1_activated = tf.nn.relu(tf.nn.bias_add(conv1,tf.Variable(tf.constant(-0.1,dtype=tf.float32,shape=[self.settings.conv1_filter_kernel]))))

        with tf.name_scope("primary_capsule"):
            # 16*16
            for i in range(self.settings.primary_capsule_size):
                self.primary_capsules.append(tf.nn.conv2d(conv1_activated,tf.Variable(tf.truncated_normal([self.settings.primary_capsule_filter,self.settings.primary_capsule_filter,self.settings.conv1_filter_kernel,self.settings.primary_capsule_dimension],stddev=1.0,dtype=tf.float32)\
                                                                                      ),strides=[1,1,1,1],padding="SAME",name="primary_capsule{0}".format(i)))
                self.primary_capsules_weights.append(tf.Variable(tf.truncated_normal([self.settings.primary_capsule_dimension,self.settings.secondary_layer_dimension],stddev=1.0,dtype=tf.float32)/
                                                                 ,name="weightij_{0}".format(i)))


        with tf.name_scope("bij"):
            bij = tf.Variable(tf.constant([self.settings.secondary_layer_size,self.settings.primary_capsule_size],value=1.0
                                                  ),name="bij")

        def squash(i):
            squared_mod = tf.square((tf.abs(i)),name="squared_mod")
            vector = i/float(tf.abs(i))
            return (squared_mod/float(1+squared_mod))*vector

        with tf.name_scope("forward_pass"):
            self.u_ = []
            self.v_ = []
            c = tf.nn.softmax(bij,dim=0) # check axis
            for i in range(self.settings.primary_capsule_size):
                self.u_.append(tf.matmul(self.primary_capsules[i],self.primary_capsules_weights))

            '''
                        sij is a vector added along all 64 primary capsules
                        will have to alter next few lines
                        maybe stack the 64 uij and do sum on particular axis
                        '''

            for i in range(self.settings.secondary_layer_size):
                temp=squash(tf.reduce_sum(tf.mul(c[i],self.u_)))
                self.v_.append(temp)

            #scalar product



# m = model()
