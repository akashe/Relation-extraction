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
    first_layer_units = 40
    second_layer_units = 40
    third_layer_units = 40


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
        pooled_outputs.append(pooled1)

    pooled_output = tf.squeeze(tf.concat(concat_dim=1,values=pooled_outputs),squeeze_dims=[-2,-1])

    with tf.name_scope("Fully_connected_nueral_network"):
        with tf.name_scope("First-layer"):
            W_first = tf.Variable(tf.truncated_normal([int(pooled_output._shape[-1]),settings.first_layer_units],mean=0.0,stddev=1.0),name = "W_First")
            b_first = tf.Variable(tf.constant(0.1,shape=[settings.first_layer_units]))
            hidden_layer_1 = tf.nn.relu(tf.matmul(pooled_output,W_first)+b_first,name="hidden_layer_1_output")

        with tf.name_scope("Second-layer"):
            W_second = tf.Variable(tf.truncated_normal([settings.first_layer_units,settings.second_layer_units],mean=0.0,stddev=1.0),name="W_Second")
            b_second = tf.Variable(tf.constant(-0.1,shape=[settings.second_layer_units]))
            hidden_layer_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer_1,W_second)+b_second,name="hidden_layer_2_output"),keep_prob=0.1)

        with tf.name_scope("Third-layer"):
            W_third = tf.Variable(tf.truncated_normal([settings.second_layer_units,settings.third_layer_units],mean=0.0,stddev=1.0),name="W_Third")
            b_third = tf.Variable(tf.constant(-0.1, shape=[settings.third_layer_units]))
            hidden_layer_3 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden_layer_1, W_third) + b_third, name="hidden_layer_3_output"),keep_prob=0.1)

    with tf.name_scope("output"):
        W_output = tf.get_variable("W_output",shape=[settings.third_layer_units,settings.num_classes],initializer=tf.contrib.layers.xavier_initializer())
        b_output = tf.Variable(tf.constant(0.1,shape=[settings.num_classes]),name="b_output")
        score = tf.nn.xw_plus_b(hidden_layer_3,W_output,b_output,name="score")

    with tf.name_scope("loss"):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=score)
        l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())
        total_loss = loss +l2_loss
        tf.scalar_summary("total_loss",total_loss)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score,1),tf.argmax(input_y,1)),"float"),name="accuracy")
        tf.scalar_summary("accuracy",accuracy)


