import tensorflow as tf
import numpy as np


######
#   The idea behind this CNN is take a full sentence and use a big window length as entire sentence can express idea.
#   Then use subsequent max pooling for feature extraction
#
#    ##############


class network_settings():
    max_sen_len = 50 # divisible by 10
    sen_split_len = 10
    number_of_layers = 3
    embedding_size = 50
    num_classes = 40
    num_filter = 10


class model():
    settings = network_settings()
    input_x = tf.placeholder(dtype=tf.int32, shape=[None, settings.max_sen_len], name="input_x")
    input_y = tf.placeholder(dtype=tf.int32, shape=[None, settings.num_classes], name="input_y")

    pooled_outputs =[]

    embedded_input = tf.expand_dims(
        tf.nn.embedding_lookup(tf.get_variable(initializer=(np.load("data/embeddings.npy").astype(dtype='float32')), name="embeddings"),
                               input_x), -1)
    with tf.name_scope("entire_sentence"):
        conv = tf.nn.conv2d(embedded_input, tf.Variable(tf.truncated_normal([settings.max_sen_len, settings.embedding_size, 1, settings.num_filter], stddev=0.1,dtype=tf.float32)),
                            strides=[1, 1, 1, 1], padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv,tf.Variable(tf.constant(0.1,shape = [settings.num_filter])),name = "relu"))

        pooled = tf.nn.max_pool(h,ksize=[1,1,1,settings.num_filter],strides=[1,1,1,1],padding='VALID',name="pool")
        pooled_outputs.append(pooled)

    with tf.name_scope("parts_of_sentence"):
        # part_length = settings.max_sen_len/settings.sen_split_len
        # for i in range(0,part_length):
        #     with tf.name_scope("sentence_part : "+str(i)):
        conv1 = tf.nn.conv2d(embedded_input, tf.Variable(
                    tf.truncated_normal([settings.sen_split_len, settings.embedding_size, 1, settings.num_filter],
                                        stddev=0.1, dtype=tf.float32)),
                                    strides=[1, settings.sen_split_len, 1, 1], padding="VALID", name="sentence_part_conv")
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.Variable(tf.constant(0.1, shape=[settings.num_filter])), name="sentence_part_relu"))
        pooled1 = tf.nn.max_pool(h1, ksize=[1, 1, 1, settings.num_filter], strides=[1, 1, 1, 1], padding='VALID',
                                name="sentence_part_pool")

