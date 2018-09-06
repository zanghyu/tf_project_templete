import tensorflow as tf
import numpy as np
import os
import argparse
from model import Model



def train(args, trained = False, total_step = 0):
    model = Model(args.min_after_dequeue, args.batch_size, args.learning_rate)


    train_dir = os.path.join('', '')
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    if not tf.gfile.Exists(train_dir):
        tf.gfile.MakeDirs(train_dir)

    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        input_data = model.input(args.filename)
        inp_batch = model.input_batch(input_data)


        output = model.inference(something)


        loss = model.loss(output, label_y)

        train_op = model.train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False))
        sess.run(init)

        if trained == True:
            ckpt = tf.train.get_checkpoint_state(os.path.join('', ''))
            print('reuse the checkpoint in %s' % os.path.join('', ''))
            saver.restore(sess, ckpt.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        threads = tf.train.start_queue_runners(sess=sess)

        for step in range(args.max_steps):
            _, loss_value = sess.run([train_op, loss])
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            virtual_step = step + total_step
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, virtual_step)

            if step % 1000 == 0 or (step + 1) == args.max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=virtual_step)
        total_step += args.max_steps
    return total_step



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument('min_after_dequeue', type=int)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('max_steps', type=int)
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()