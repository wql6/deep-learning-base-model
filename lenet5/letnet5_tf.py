# -*- coding: utf-8 -*-
"""
__author__ = 'Alex wu'
__version__ = '1.0'
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    c_conv1 = tf.nn.conv2d(x_image, filter=weights([5, 5, 1, 32]), strides=[1, 1, 1, 1], padding='VALID')
    h_conv1 = tf.nn.relu(tf.nn.bias_add(c_conv1, bias([32])))
    p_conv1 = tf.nn.avg_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    c_conv2 = tf.nn.conv2d(p_conv1, filter=weights([5, 5, 32, 64]), strides=[1, 1, 1, 1], padding='VALID')
    h_conv2 = tf.nn.relu(tf.nn.bias_add(c_conv2, bias([64])))
    p_conv2 = tf.nn.avg_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    shape = p_conv2.get_shape().as_list()
    nodes = shape[1] * shape[2] * shape[3]  # nodes=3136
    reshaped = tf.reshape(p_conv2, [-1, nodes])


    f_conv3 = tf.layers.Flatten()(reshaped)
    # 或者是 f_conv3 = tf.layers.flatten(inputs=reshaped)
    d_conv3_1 = tf.layers.dense(inputs=f_conv3, units=120, activation='sigmoid')
    d_conv3_2 = tf.layers.dense(inputs=d_conv3_1, units=84, activation='sigmoid')
    d_conv3_3 = tf.layers.dense(inputs=d_conv3_2, units=10, activation='softmax')



    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(d_conv3_3, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # important step
    sess.run(tf.global_variables_initializer())
    # #################优化神经网络##################################
    # print('shape:', sess.run(p_conv1))

    saver = tf.train.Saver()
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # print(batch_xs.shape)
        # print(batch_ys.shape)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            # print(sess.run(prediction,feed_dict={xs:batch_xs}))
            y_pre = sess.run(d_conv3_3, feed_dict={x: mnist.test.images})
            correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(mnist.test.labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print('accuracy:', result)
    saver.save(sess, 'my_net.ckpt')


if __name__ == '__main__':
    main()

