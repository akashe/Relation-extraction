import tensorflow as tf
import numpy as np



class network_settings():
    max_sen_len = 50  # divisible by 10
    sen_split_len = 10
    number_of_layers = 3
    embedding_size = 50
    num_classes = 15
    num_filter = 10
    first_layer_units = 40
    second_layer_units = 40
    third_layer_units = 40
    batch_size = 100
    epochs = 50
    test_batch_size = 1000
    initial_filter = 2
    number_of_filter_in_1_layer =2
    use_1by1 = True
    filter_size_across_layers = 10


class model():
    def __init__(self):
        self.settings = network_settings()
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.max_sen_len], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.settings.num_classes], name="input_y")

        pooled_outputs = []

        embedded_input = tf.expand_dims(
            tf.nn.embedding_lookup(tf.get_variable(initializer=(np.load("../data/embeddings.npy").astype(dtype='float32')),
                                                   name="embeddings"),
                                   self.input_x), -1)

        filters = self.get_filter_size(self.settings.number_of_filter_in_1_layer,self.settings.initial_filter,self.settings.max_sen_len)

        ## Possibility of using same weights across

        with tf.name_scope("conv-"):
            ip = embedded_input
            ip_filter_depth_size = 1
            ip_filter_width_size = self.settings.embedding_size
            op_filter_size = self.settings.filter_size_across_layers
            for filter in filters:
                a= 0
                each_filter_op = []
                for i in filter:
                    each_filter_op.append(tf.nn.conv2d(ip,tf.Variable(tf.truncated_normal(shape=[i,ip_filter_width_size,ip_filter_depth_size,op_filter_size]\
                                                                                          ,stddev = 0.1,dtype = tf.float32)),strides=[1,1,ip_filter_width_size,1],padding="SAME"\
                                                       ,name="{0}".format(i)))
                    a+=1
                if self.settings.use_1by1:
                    each_filter_op.append(tf.nn.conv2d(ip,tf.Variable(tf.truncated_normal(shape=[1,ip_filter_width_size,ip_filter_depth_size,op_filter_size]\
                                                                                      ,stddev = 0.1,dtype = tf.float32)),strides=[1,1,ip_filter_width_size,1],padding="SAME"\
                                                   ,name="{0}".format(i)))
                    a+=1
                    ## LAter check:
                    ## Can I just use the previos layer?
                    ## How to make it addition instead of so much multiplication
                ip = tf.concat(each_filter_op,axis=-2)
                ip_filter_depth_size = op_filter_size
                ip_filter_width_size = a

        logits = tf.nn.conv2d(ip,tf.Variable(tf.truncated_normal(shape=[self.settings.max_sen_len,ip_filter_width_size,ip_filter_depth_size,self.settings.num_classes]\
                                                                                      ,stddev = 0.1,dtype = tf.float32)),strides=[1,self.settings.max_sen_len,ip_filter_width_size,ip_filter_depth_size],padding="SAME"\
                                                   ,name="{0}".format(i))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.squeeze(logits),labels=self.input_y)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),tf.argmax(self.input_y,1)),'float'),name='accuracy')

        print("model ready")

    def get_filter_size(self,nofilter1layer,initial,max_len):  # include stride also later
        filters = []
        i = initial
        j = 0
        while(not j>=max_len):
            a= []
            if i==initial:
                j= initial
            else:
                j = i + j -1
            for k in range(nofilter1layer):
                a.append(i)
                i+=1
            filters.append(a)
        return filters

    def get_pad_value(self,filter_size):
        pass

    def conv_layer(self,embedded_input):
        tf.nn.conv2d(embedded_input, tf.Variable(tf.truncated_normal(
            [self.settings.max_sen_len, self.settings.embedding_size, 1, self.settings.num_filter], stddev=0.1,
            dtype=tf.float32)),
                     strides=[1, 1, 1, 1], padding="VALID", name="conv")



m = model()
