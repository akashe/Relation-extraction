import tensorflow as tf
from bidirectionalLSTM_multiwindow import model
from bidirectionalLSTM_multiwindow import Graph_Handler
from process_data import Dataset
# import tqdm

##
#   In this file seperate graph for train and test
#   seperate graphs for train test and inference
#   TODO: next version to have config and flags and
#   #

def main():
    train_data = Dataset("train",False) # Set true while changing num of classes
    test_data = Dataset("test",False) # This will result in disproportionate distribution of examples in test and train

    train_graph = tf.Graph()
    test_graph = tf.Graph()

    with train_graph.as_default():
        train_G = Graph_Handler("save","summary")
        init = tf.global_variables_initializer()

    train_sess = tf.Session(graph=train_graph)
    test_sess = tf.Session(graph=test_graph)

    with test_graph.as_default():
        test_G = Graph_Handler("save","summary")
        # init = tf.global_variables_initializer()
        test_G.initialize(test_sess)

    train_sess.run(init)


    for i,data in enumerate(train_data.get_batch()):
        global_step = train_sess.run(train_G.global_step) + 1
        if i % train_G.summary_period == 0 :
            accuracy = train_G.run(train_sess,data=data,summary=True,global_step=global_step)
        else:
            accuracy = train_G.run(train_sess, data=data, summary=False)

        print("Accuracy at global step {} is {}".format(i, accuracy))

        if i% train_G.eval_period == 0 and i!=0:
            with train_graph.as_default():
                train_G.save(train_sess)
            with test_graph.as_default():
                # test_G.initialize(test_sess)
                test_G.restore(test_sess)
            test_acc = test_G.eval(test_sess,dataset=test_data)
            print(" Test Accuracy after global step {} is {}".format(train_G.global_step, test_acc))
            #TODO add part to save best model


if __name__ == '__main__':
    main() # tf.app??