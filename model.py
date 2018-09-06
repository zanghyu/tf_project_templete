import tensorflow as tf
import numpy as np
import os

###  Define some constants in here
BATCH_SIZE = 128

class Model(object):
    def __init__(self, min_after_dequeue = 10000, batch_size = 128, learning_rate = 0.01):
        self.min_after_dequeue = min_after_dequeue
        self.batch_size = batch_size
        self.learning_rate = learning_rate



    def input(self,filename):
        '''
        load the input data from file
        :param filename:
        :return:
        '''

        file = os.path.join('some_path','%s'%filename)
        if os.path.exists(filename):
            print('%s is found' % (filename))
        else:
            raise ValueError(" %s is not found" % filename)

        #### load data from file


        input_data = []
        return input_data


    def input_batch(self, input_data):
        '''
        change the input data to batch
        :param input_data:
        :param batch_size:
        :return:
        '''
        min_after_dequeue = self.min_after_dequeue
        capacity = min_after_dequeue + 3 * self.batch_size

        inp_batch = tf.train.shuffle_batch([input_data],batch_size=self.batch_size,
                                           min_after_dequeue=self.min_after_dequeue,
                                           capacity=capacity,enqueue_many=True)
        ### this function **tf.train.shuffle_batch** can read from file use multithread,
        ### and capacity must be larger than min_after_dequeue.


        inp_batch = tf.cast(inp_batch, tf.float32)

        return inp_batch

    def inference(self, something):
        '''
        the core model, from input to output

        :param something:
        :return:
        '''
        with tf.variable_scope('first_layer') as scope:
            first_layer = tf.contrib.layers.fully_connected(inputs=__,num_outputs=__,activation_fn=tf.nn.relu)

        with tf.variable_scope('second_layer') as scope:
            second_layer = __

        ...

        with tf.variable_scope('final_layer') as scope:
            final_layer = __

        return  final_layer

    def loss(self, final_layer_result, label_y):
        '''
        define what loss we should use
        :param final_layer_result:
        :param label_y:
        :return:
        '''
        l2_loss = tf.nn.l2_loss(final_layer_result - label_y) / BATCH_SIZE
        tf.add_to_collection('losses', l2_loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('total_loss', total_loss)
        return total_loss

    def train(self, total_loss, global_step):
        '''
        define optimizer and some other training ops

        :param total_loss:
        :param global_step:
        :return:
        '''


        apply_gradients_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=global_step,
            learning_rate=self.learning_rate,
            optimizer='Adam')

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.control_dependencies([apply_gradients_op]):
            train_op = tf.no_op(name='train')

        return train_op