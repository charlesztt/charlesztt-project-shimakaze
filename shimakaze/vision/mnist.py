import numpy as np
import tensorflow as tf


class MNIST:
    def __init__(self, path=None, finetune=False):
        '''
        path: None finetune: False => Train from scratch
        path: None finetune: True => Error
        path: xxx finetune: False => Test mode
        path: xxx finetune: True => Train with finetune
        :param path:
        :param finetune:
        '''
        if path is None:
            if finetune is False:
                print("Train a model from scratch")
                self.train_mode=True
            else:
                raise Exception("You can't finetune without a starting point")
        else:
            if finetune is False:
                print("Test a model")
                self.train_mode=False
            else:
                print("Finetune a model with pretrained model")
                self.train_mode=True
        self.finetune=finetune
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        self.vlist = list()

        # Special for this dataset: images are in a flat form.
        image = tf.reshape(self.x, [-1, 28, 28, 1])

        # conv_1
        self.conv_1 = self._conv_layer(image, [5,5,1,32], [32], "conv_1")
        self.pool_1 = self._max_pool(self.conv_1)

        # conv_2
        self.conv_2 = self._conv_layer(self.pool_1, [5, 5, 32, 64], [64], "conv_2")
        self.pool_2 = self._max_pool(self.conv_2)

        # fc_1
        self.fc_1 = self._fc_layer(tf.reshape(self.pool_2, [-1, 7*7*64]), [7*7*64, 1024], [1024], "fc_1")

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.fc_1_dropout = tf.nn.dropout(self.fc_1, self.keep_prob)

        # fc_2
        self.fc_2 = self._fc_layer(self.fc_1_dropout, [1024, 10], [10], "fc_2")

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.fc_2))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.fc_2, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.saver = tf.train.Saver(self.vlist)

        if path != None:
            new_saver = tf.train.Saver(self.vlist)
            new_saver.restore(self.sess, "./%s"%path)


    def train(self, mnist, expected_steps=1000):
        if self.train_mode is False:
            raise Exception("Sorry I can't train it...")
        if self.finetune is True:
            to_do_var_list = list()
            var_list = tf.global_variables()+tf.local_variables()
            for one_var in var_list:
                if self.sess.run(tf.is_variable_initialized(one_var)):
                    pass
                else:
                    to_do_var_list.append(one_var)
            self.sess.run(tf.variables_initializer(to_do_var_list))
        else:
            self.sess.run(tf.global_variables_initializer())

        for i in range(expected_steps):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.train_step.run(session=self.sess, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        print("test accuracy %g" % self.accuracy.eval(session=self.sess, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))

    def save_model(self, path):
        self.saver.save(self.sess, path)
        print("model saved to %s"%path)

    def _fc_layer(self, bottom, weight_shape, bias_shape, name):
        with tf.variable_scope(name):
            weight = self._weight_variable(weight_shape, "weight")
            self.vlist.append(weight)
            bias = self._bias_variable(bias_shape, "bias")
            self.vlist.append(bias)
            return tf.nn.relu(tf.matmul(bottom, weight)+bias)

    def _conv_layer(self, bottom, weight_shape, bias_shape, name):
        with tf.variable_scope(name):
            weight = self._weight_variable(weight_shape, "weight")
            self.vlist.append(weight)
            bias = self._bias_variable(bias_shape, "bias")
            self.vlist.append(bias)
            return tf.nn.relu(tf.nn.conv2d(bottom, weight, [1, 1, 1, 1], padding='SAME') + bias)


    def _max_pool(self, bottom):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name):
        # initial = tf.constant(0.1, shape=shape)
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)
