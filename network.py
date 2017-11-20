import tensorflow as tf
import numpy as np


######
#   The idea behind this CNN is take a full sentence and use a big window length as entire sentence can express idea.
#   Then use subsequent max pooling for feature extraction
#
#    ##############


class network_settings():
    max_sen_len = 50
    number_of_layers = 3
    embedding_size = 50
    num_classes = 40


class model():
    settings = network_settings()
    input_x = tf.placeholder(dtype=tf.int32,shape=[None, settings.max_sen_len],name="input_x")
    input_y = tf.placeholder(dtype=tf.int32,shape=[None, settings.num_classes], name= "input_y")

    embedded_input = tf.nn.embedding_lookup(tf.get_variable(initializer=(np.load("data/embeddings.npy")),name="embeddings"),input_x)


