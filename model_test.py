import tensorflow as tf
import numpy as np
import os
from model import Model

def perdict(sess, model, something):
    checkpoint_dir = os.path.join('', '')
    inp = tf.placeholder(tf.float32, shape=input_shape, name='inp')

    output = model.inference(something)

    saver = tf.train.Saver(tf.trainable_variables())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print('load the checkpoint: %s' % checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    return inp, output


def test(args):
    model = Model()


    with tf.Graph().as_default() as g:

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            inp, output = perdict(sess, model, something)



            sess.run(output, feed_dict={inp: inp_data})




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("", type=int)
    args = parser.parse_args()

    test(args)


if __name__ == '__main__':
    main()