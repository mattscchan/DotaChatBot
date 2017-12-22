from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss


def _parse_function(example_proto):
    features = {
        "context": tf.FixedLenFeature([], tf.float32),
        "target": tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features['context'], parsed_features['target']

def create_dataset(name):
    data = tf.data.TFRecordDataset(name)
    data = data.map(_parse_function, num_parallel_calls=8)
    data = data.batch(100)
    data = data.shuffle(100000)
    data = data.prefetch(100000)
    train_iterator = data.make_initializable_iterator()
    next_el = train_iterator.get_next()

    return next_el, train_iterator

def word2vec(batch_gen, iterator):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # Step 2: define weights. In word2vec, it's actually the weights that we care about

    with tf.name_scope('embedding_matrix'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), 
                            name='embed_matrix')

    # Step 3: define the inference
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

        # Step 4: construct variables for NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
                                                    stddev=1.0 / (EMBED_SIZE ** 0.5)), 
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

        # define loss function to be NCE loss function
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                            biases=nce_bias, 
                                            labels=target_words, 
                                            inputs=embed, 
                                            num_sampled=NUM_SAMPLED, 
                                            num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={name: ['./data/100k_skipgrams.tfrecords']})

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./data/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            batch = sess.run(batch_gen)
            loss_batch, _ = sess.run([loss, optimizer], 
                                    feed_dict={center_words: batch[0], target_words: batch[1]})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main(args):
    name = tf.placeholder(tf.string, shape=[None])
    batch_gen, iterator = create_dataset(name)
    word2vec(batch_gen, iterator)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames')
    args = parser.parse_args()
    main(args)