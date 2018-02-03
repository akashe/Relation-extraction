import tensorflow as tf
import numpy as np
import math

######
#   Capsule network for relation extraction
#   Current version memory heavy and slow
#   Model not converging!
#    ##############

# Notes : First run Obesrvation : model very slow even for just cpu run

# TODO upadte the model to newer tensorflow verisons (use norm and slicing operations)

class network_settings():
    max_sen_len = 50  # divisible by 10
    conv1_filter_shape = 20
    conv1_filter_kernel = 32
    embedding_size = 50
    num_classes = 15
    num_filter = 10
    batch_size = 100
    epochs = 50
    test_batch_size = 1000
    primary_capsule_size = 64
    primary_capsule_dimension =4
    primary_capsule_filter = 10
    primary_capsule_width = 16
    weights_i_j = primary_capsule_dimension
    secondary_layer_dimension = 5
    secondary_layer_size = num_classes

class model():
    def __init__(self):
        self.settings = network_settings()
        self.primary_capsules = []
        self.primary_capsules_weights = []
        self.u_ = []
        self.uv = []
        self.s_ = []
        self.v_ = []
        self.v_magnitude= []
        self.v_magnitude_reshape = []
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.max_sen_len], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_classes], name="input_y")
        # check wether list are re-initialised or not?? Does moving them under scope makes them part of graph?
        # look more into segmentation
        self.temp_bij = []
        pooled_outputs = []

        bij = tf.Variable(
            tf.constant(shape=[self.settings.primary_capsule_size, self.settings.secondary_layer_size], value=1.0,
                        dtype=tf.float32
                        ), name="bij")

        embedded_input = tf.expand_dims(
            tf.nn.embedding_lookup(tf.get_variable(initializer=(np.load("data/embeddings.npy").astype(dtype='float32')),
                                                   name="embeddings"),
                                   self.input_x), -1)

        with tf.name_scope("conv1"):
            conv1 = tf.nn.conv2d(embedded_input,tf.Variable(tf.truncated_normal([self.settings.conv1_filter_shape,self.settings.conv1_filter_shape,1,self.settings.conv1_filter_kernel],stddev=0.1,dtype=tf.float32)),strides=[1,2,2,1],padding="SAME")
            conv1_activated = tf.nn.relu(tf.nn.bias_add(conv1,tf.Variable(tf.constant(-0.1,dtype=tf.float32,shape=[self.settings.conv1_filter_kernel]))))

        with tf.name_scope("primary_capsule"):
            # 16*16
            for i in range(self.settings.primary_capsule_size):
                ## Should I put Relu activation over these??
                self.primary_capsules.append(tf.nn.conv2d(conv1_activated,tf.Variable(tf.truncated_normal([self.settings.primary_capsule_filter,self.settings.primary_capsule_filter,self.settings.conv1_filter_kernel,self.settings.primary_capsule_dimension],stddev=1.0,dtype=tf.float32)\
                                                                                      ),strides=[1,1,1,1],padding="VALID",name="primary_capsule{0}".format(i)))
                self.primary_capsules_weights.append(tf.Variable(tf.truncated_normal([self.settings.primary_capsule_dimension,self.settings.secondary_layer_dimension,self.settings.secondary_layer_size],stddev=1.0,dtype=tf.float32),name="weightij_{0}".format(i)))

                self.u_.append(tf.reshape(tf.matmul(tf.reshape(self.primary_capsules[i],[-1,self.settings.primary_capsule_dimension]),tf.reshape(self.primary_capsules_weights[i],[self.settings.primary_capsule_dimension,-1])),[-1,self.settings.primary_capsule_width,self.settings.primary_capsule_width,self.settings.secondary_layer_dimension,self.settings.secondary_layer_size]))
                # self.u_.append(tf.matmul(self.primary_capsules[i], tf.reshape(self.primary_capsules_weights[i],
                #                                                               [self.settings.primary_capsule_dimension,
                #                                                                -1])))

                ## Matmul needs matrices of rank greater than 2.. Check why weight vector isnt of rank >2

        with tf.name_scope("bij"):

            c = tf.nn.softmax(bij, dim=-1)

        def squash(s):
            # x = tf.norm(s) Since lack of this sch options in 0.12 doing
            # x = tf.reduce_sum(s)
            # This part sucks coz we are calculating magnitude for the entire batch!
            x = tf.reduce_logsumexp(s,axis=[1,2,3])
            x2 = tf.square(x)
            v_ = tf.divide(x2,(1+x2))
            return v_,tf.mul(v_,tf.div(s,x))


        with tf.name_scope("forward_pass"):
            for i in range(self.settings.primary_capsule_size):
                ## Am I consuming too much energy by creating this again and again??
                cij = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(c[i],0),[self.settings.secondary_layer_dimension,1]),0),[self.settings.primary_capsule_width,1,1]),0),[self.settings.primary_capsule_width,1,1,1])
                ## I didn't have to do this I could have simply done tf.mul with c[i], it would still work!
                # self.uv.append(tf.reshape(tf.mul(self.u_[i],cij),[-1,self.settings.secondary_layer_size]))
                self.uv.append(tf.mul(self.u_[i], cij))

                for j in range(self.settings.secondary_layer_size):
                    if len(self.s_)!=j:
                        self.s_[j] = tf.add(self.s_[j],self.uv[i][:,:,:,:,j])
                        ### Do slicing operations take more time?
                        ######## Use tf.split here
                    else:
                        self.s_.append(self.uv[i][:,:,:,:,j])

            # v_major = tf.concat(0,self.uv)
            for i in range(self.settings.secondary_layer_size):
                j_,k_ =squash(self.s_[i])
                self.v_.append(k_)
                self.v_magnitude.append(j_)

            for i in range(self.settings.primary_capsule_size):
                for j in range(self.settings.secondary_layer_size):
                    self.temp_bij.append(tf.reduce_mean(tf.matmul(tf.reshape(self.u_[i][:, :, :, :, j], [-1, self.settings.secondary_layer_dimension]),tf.transpose(tf.reshape(self.v_[j], [-1, self.settings.secondary_layer_dimension])))))

        bij = tf.reshape(tf.add(tf.reshape(bij,[-1]),tf.convert_to_tensor(self.temp_bij)),[self.settings.primary_capsule_size,self.settings.secondary_layer_size])
                    # self.a =tf.matmul(tf.reshape(self.u_[i][:, :, :, :, j], [-1, self.settings.secondary_layer_dimension]),tf.transpose(tf.reshape(self.v_[j], [-1, self.settings.secondary_layer_dimension])))
                    # tf.assign_add(bij[i][j],tf.reduce_mean(self.a))
                    ## Are these constants kept sperately in memorty
        with tf.name_scope("loss"):
            # Methods:
            #   a) fully connected network
            #   b) simple matrix multiplication
            #   c) Use magnitude values

            ###### Reshape logits ######

            self.v_magnitude_reshape = tf.transpose(tf.convert_to_tensor(self.v_magnitude))



            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.v_magnitude_reshape)
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                  weights_list=tf.trainable_variables())
            self.total_loss = tf.reduce_mean(self.loss) + self.l2_loss
            tf.scalar_summary("total_loss", self.total_loss)

            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(self.v_magnitude_reshape, 1), tf.argmax(self.input_y, 1)), "float"), name="accuracy")
                tf.scalar_summary("accuracy", self.accuracy)

            print ("model ready")

m = model()
