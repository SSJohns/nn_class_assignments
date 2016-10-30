"""Iris example with Tensorflow.
"""
from __future__ import print_function, division

import sys
import csv
import numpy as np
from random import shuffle
import tensorflow as tf


class Model(object):

    def __init__(self, is_training=True, batch_size=10):
        self.is_training = is_training
        self.batch_size = batch_size

        # Data placeholders, i.e. the input gate where data come in
        self.x = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.y = tf.placeholder(tf.int64, [self.batch_size])

        # Model variables
        # As in the irisnn.py example, we use only 1 layer with 16 units, and
        # sigmoid as activation function.
        with tf.variable_scope('hidden'):
            # This variable will make the below weight and bias under the /hidden
            # scope. Thus their names will become hidden/W_h and hidden/b_h
            W_h_1 = tf.get_variable('W_h_1', [4, 32])
            b_h_1 = tf.get_variable('b_h_1', [32])

            W_h_2 = tf.get_variable('W_h_2', [32, 16])
            b_h_2 = tf.get_variable('b_h_2', [16])

        # Then the logit layer
        with tf.variable_scope('logit_layer'):
            W_l = tf.get_variable('W_l', [16, 3])
            b_l = tf.get_variable('b_l', [3])

        # Let's glue them all together
        # First the hidden layer, of course instead of tf.add, we can use +
        hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.x, W_h_1), b_h_1))
        hidden_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_1, W_h_2), b_h_2))

        # Logit layer
        logits = tf.matmul(hidden_2, W_l) + b_l
        # Let's also calculate the probabilities
        self.probs = tf.nn.softmax(logits)

        # Now the loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, self.y)
        self.loss = tf.reduce_mean(cross_entropy)

        # Now, if our model is for training, we need to get the gradients, and
        # define the training strategy.
        if self.is_training:
            # Learning rate
            # We don't learn learning rate
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.loss)
        else:
            # If we're evaluating, we only need to calculate the prediction
            # accuracy
            correct = tf.stop_gradient(tf.nn.in_top_k(logits, self.y, 1))
            self.accuracy = tf.reduce_sum(
                tf.to_float(correct)) / tf.to_float(self.batch_size)


def importData():
    # dict converting the species to ints
    species = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    # Our data is taken from here https://archive.ics.uci.edu/ml/datasets/Iris
    # afaik, all UCI dataset has description of its format, (check the link above)
    # For this dataset, it's a csv file whose each line = data of 1 sample
    # in format: feature1, feature2, feature3, feature4, class_name
    # While the features are numbers (in cm), the class name is a string
    # We have only 3 classes, thus we index them as 0, 1, and 2 using above
    # dictionary.
    data = []

    # Instead of opening a file then closing it (a must), use ```with``` so that
    # the file is automatically closed after we're done processing.
    with open('./iris.dat', 'r') as f:
        # create a csv reader object
        reader = csv.reader(f)
        for line in reader:
            # Each line is an array of values, say [4.3, 2.5, 3.4, 5.2, 'Iris-setosa']
            # i.e., the first four features (sepal & petal lengths & widths), and
            # the corresponding class. If there's an empty line, ```line``` here
            # is an empty list, ignore it
            if line:
                data.append(line)

    shuffle(data)
    # you know the array sizes, so you can hardcode the shape here
    # if you didn't, you could use len() to figure out how many you had
    x = np.zeros((150, 4))
    y = np.zeros(150)

    # You need the data as well as the array indices, so enumerate is helpful
    # here
    for i, line in enumerate(data):
        x[i] = np.asarray(line[:4])  # cast the list into an array
        # a little layered, but use the dict to set the correct entry to True
        y[i] = species[line[4]]

    return x, y


def train(num_epochs, batch_size):
    print("Train for {} epochs, with batch size {}".format(
        num_epochs, batch_size))

    # Get data
    x, y = importData()
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Split into train/dev
    split = int(x.shape[0] * 0.8)
    train_x, train_y = x[:split], y[:split]
    eval_x, eval_y = x[split:], y[split:]

    # Just some training information
    num_batches_per_epoch = len(train_x) // batch_size
    total_batches = num_batches_per_epoch * num_epochs

    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_normal_initializer(-0.1, 0.1)
        with tf.variable_scope('iris', reuse=False, initializer=initializer):
            train_model = Model(is_training=True, batch_size=batch_size)
        with tf.variable_scope('iris', reuse=True, initializer=initializer):
            eval_model = Model(is_training=False, batch_size=len(eval_x))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        train_indices = np.arange(len(train_x))
        for e in xrange(num_epochs):
            # It's a good practice to shuffle training data before each epoch
            np.random.shuffle(train_indices)
            train_x = train_x[train_indices]
            train_y = train_y[train_indices]

            epoch_loss = 0.0
            for b in xrange(num_batches_per_epoch):
                batch_x = train_x[b * batch_size: (b + 1) * batch_size]
                batch_y = train_y[b * batch_size: (b + 1) * batch_size]
                feed = {
                    train_model.x: batch_x,
                    train_model.y: batch_y
                }
                loss, _ = sess.run(
                    [train_model.loss, train_model.train_op], feed)
                epoch_loss += loss / num_batches_per_epoch

            # We evaluate after each epoch
            eval_feed = {
                eval_model.x: eval_x,
                eval_model.y: eval_y
            }
            accuracy = sess.run(eval_model.accuracy, eval_feed)
            print("Epoch {}/{}, loss={}, accuracy={}".format(e + 1,
                                                             num_epochs, epoch_loss, accuracy))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(
            'Invalid arguments. Usage: python iris_tensorflow.py <num_epochs> <train_batch_size>')

    train(int(sys.argv[1]), int(sys.argv[2]))
