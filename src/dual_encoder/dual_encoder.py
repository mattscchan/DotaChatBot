'''
@authors
William Le
Matthew Chan

@date
2017-12-21

@version
Python 3.5.4
TensorFlow 1.4.0
'''

import csv
import time
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

SEED = int('0xCAFEBABE', 16)

# For reproducibility purposes
np.random.seed(SEED)
tf.set_random_seed(SEED)

## Hyperparameters ##
Hyper = namedtuple('Hyper', [
    'initializer',      #tensorflow function
    'optimizer',        #tensorflow function
    'cell_type',        #string
    'max_timesteps',    #floats
    'hidden_size',
    'num_layers',
    'batch_size',
    'dropout_prob',
    'initial_lr',
    'decay_rate',
    'decay_steps',
    ])

## Constants ##
Const = namedtuple('Const', [
    'embedding_size',
    'embedding_vocab',
    'num_classes',
    ])

## Initialize hyperparameters and constants ##
def create_params(params_file, number=-1):
    '''
    Reads a file containing the comma delimited hyperparameters, then constants on one line
    First two arguments are tensorflow functions (initializer, optimizer)
    Second is a string (cell_type)
    Last three are constants
    '''
    with open(params_file, newline='') as f:
        for i, line in enumerate(csv.reader(f)):
            if i == number:
                break
            else:
                pass
    print(line)
    params = [eval(f) for f in line[:2]]
    params.append(line[2])
    params.extend(float(v) for v in line[3:])
    h = Hyper(*params[:-3])
    c = Const(*params[-3:])
    print('Hyperparameters:', h)
    print('Constants:', c)
    return h, c

## Architecture ##
def create_rnn(cell_type, hidden_size, dropout_prob=None, num_layers=1, batch_size=1):
    '''
    Intialize an RNN unit used for encoding both the context and the response vectors.

    Parameters
    ----------
    cell_type:      string, one of {'rnn', 'gru', 'lstm'}
    hidden_size:    int, number of hidden nodes of the RNN, also the output size
    dropout_prob:   int, probability of zeroed out node
    num_layers:     int, number of hidden layers for the RNN
    batch_size:     int, size of training batch

    Outputs
    -------
    an instance of tf.nn.rnn_cell.RNNCell
    a initial RNN state tensor of shape (batch_size, hidden_size)

    '''
    with tf.variable_scope('RNN'):
        cell_unit = {'rnn': tf.nn.rnn_cell.BasicRNNCell,
                'gru': tf.nn.rnn_cell.GRUCell,
                'lstm': tf.nn.rnn_cell.LSTMCell}

        cell = cell_unit[cell_type.lower().strip()](hidden_size)
        if dropout_prob:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-dropout_prob)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        # Initialize the state tensor
        initial_state = cell.zero_state(batch_size, tf.float32)

        return cell, initial_state

def create_dual_encoder(context, response, cell, state, weights, bias, initializer):
    '''
    Creates a single RNN instance used to encode both the context and the response vectors.
    This is due to using a weight tied architecture for the dual encoder.
    Multiplies the output state of both encodings and multiples them together with a learned weight matrix.

    Parameters
    ----------
    context:        tensor(batch_size, max_timesteps, embedding_dimensions)
    response:       tensor(batch_size, max_timesteps, embedding_dimensions)
    cell:           RNNCell instance with the same batch_size
    state:          intial state tensor of cell
    weights:        tensor(batch_size, max_timesteps, max_timesteps) for combining context and response outputs
    bias:           tensor(batch_size)
    initializer:    weights initializer for the RNN

    Outputs
    -------
    tensor(batch_size) corresponding to the logits = [(c^T)Mr] + b
    '''
    with tf.variable_scope('RNN', initializer=initializer()):
        # Length of the unpadded portion of the inputs: a time_step value <= max_timesteps
        context_length = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(context), 2)), 1)
        response_length = tf.reduce_sum(tf.sign(tf.reduce_max(tf.abs(response), 2)), 1)

        # context_state is a 2 tuple containing the (hidden_state, output)
        # in her paper, Pineau uses hidden state
        _, context_state = tf.nn.dynamic_rnn(cell, context, context_length, state, dtype=tf.float32)
        _, response_state = tf.nn.dynamic_rnn(cell, response, response_length, state, dtype=tf.float32)
        print('Context:', context.shape)
        print('Response:', response.shape)

        context_state = tf.reshape(context_state[0], [-1, 1, context.shape[1]])
        response_state = tf.reshape(response_state[0], [-1, 1, response.shape[1]])

        print('RNN context output/state:', context_state.shape, context_state.shape)
        print('RNN response output/state:', response_state.shape, response_state.shape)

        generated_context = tf.matmul(weights, response_state, transpose_b=True)
        similarity = tf.reshape(tf.matmul(context_state, generated_context), [-1])
        logits = tf.add(similarity, bias, name='logits')
        print('Logits:', similarity.shape, logits.shape)

        return logits

def create_performance(labels, logits, optimizer, lr, decay_rate, decay_steps, global_step):
    '''
    The TensorFlow training and accuracy operation
    '''
    with tf.name_scope("loss"):
        cross = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross, name="loss")

    with tf.name_scope("train"):
        if not (decay_rate or decay_steps):
            decay_rate = 1.0
            decay_steps = 1

        learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate, staircase=True)
        optimizer = optimizer(learning_rate)
        training = optimizer.minimize(loss, global_step=global_step)

    with tf.name_scope("eval"):
        pred_onehot = tf.nn.softmax(logits)
        pred = tf.argmax(pred_onehot, axis=1)
        print('Logits:', logits.shape)
        print('Predictions:', pred.shape)
        correct = tf.equal(pred, labels)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return training, accuracy

def train(h, c, filenames_train, filenames_valid, training_op, accuracy_op, save_path, saved, epochs=1, batch_size=32):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        with open('training.log', 'a') as log:
            time_start = time.time()
            if saved:
                saver.restore(sess, save_path + '.ckpt')
            else:
                sess.run(tf.global_variables_initializer())

            element_train, iterator_train = create_dataset(filenames_train, 
                    parse_function=parse_JSON, 
                    batch_size=batch_size, 
                    num_epochs=epochs)
            element_valid, iterator_valid = create_dataset(filenames_valid, 
                    parse_function=parse_JSON, 
                    batch_size=batch_size, 
                    num_epochs=epochs)

            for epoch in range(epochs):
                sess.run(iterator_train.initalizer, feed_dict={filenames_train: filenames_train})
                while True:
                    try:
                        batch_train = sess.run(element_train)
                    except tf.errors.OutOfRangeError:
                        break # Catch error when generator runs out after each epoch
                    else:
                        # Training the model
                        sess.run(training_op, feed_dict={
                            x_context:batch[0],
                            x_response:batch[1],
                            y:batch[2]})

                        # Get trainig accuracy
                        accuracy_train = accuracy_op.eval(session=sess, feed_dict={
                            x_context:batch[0],
                            x_response:batch[1],
                            y:batch[2]})
                        print('Training accuracy: %2.2f' % accuracy_train*100, end='\r', flush=True)
                # Validating after every epoch
                print('Training accuracy: %2.2f' % accuracy_train*100)
                sess.run(iterator_valid.initalizer, feed_dict={filenames_valid: filenames_valid})
                while True:
                    try:
                        batch_valid = sess.run(element_valid)
                    except tf.errors.OutOfRangeError:
                        break
                    except:
                        accuracy_valid = accuracy_op.eval(session=sess, feed_dict={
                            x_context:batch[0],
                            x_response:batch[1],
                            y:batch[2]})
                        print('Validating accuracy: %2.2f' % accuracy_valid*100)
                saved_at = saver.save(sess, save_path + str(epoch) + '.ckpt')
                log.write(str(c) + ',' + str(h) + 'epoch:' + str(epoch))
                log.write('t_acc:' + str(accuracy_train) + 'v_acc:' + str(accuracy_valid) + '\n')

def parse_JSON(example):
    parsed_ex = tf.decode_json_example(example)
    return parsed_ex['context'], parsed_ex['next_utt'], parsed_ex['label']

def create_dataset(filenames, parse_function, num_parallel_calls=1, batch_size=32,  shuffle_buffer=10000, num_epochs=1):

    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)

    if num_epochs < 0:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)

        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.padded_batch(batch_size, [3, None])
        dataset = dataset.prefetch(10000)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    return next_element, iterator

def get_saved_path(save_path):
    pass

def parse_JSON(example):
    parsed_ex = tf.decode_json_example(example)
    obj_ex = tf.parse_single_example(parsed_ex,
    {
        'context': tf.VarLenFeature(tf.int64),
        'next_utt': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    print('DEBUG')
    return obj_ex['context'], obj_ex['next_utt'], obj_ex['label']

def create_dataset(filenames, parse_function, num_parallel_calls=1, batch_size=32,  shuffle_buffer=10000, num_epochs=1):
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)

    if num_epochs < 0:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)

        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.padded_batch(batch_size, [3, None])
        dataset = dataset.prefetch(10000)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    return next_element, iterator

def main(args):
    # Load parameters from file
    print('Loading parameters')
    hyper, const = create_params('src/dual_encoder/params.csv', args.param_line)
    current_epoch= tf.Variable(0, trainable=False)

    # Inputs
    print('Defining inputs')
    filenames_train = tf.placeholder(tf.string, shape=None)
    filenames_valid = tf.placeholder(tf.string, shape=None)
    embeddings = tf.placeholder(tf.float32, shape=([const.embedding_vocab, const.embedding_size]), name='embeddings')
    x_context = tf.placeholder(tf.int32, shape=([None, hyper.max_timesteps ]), name='x_context')
    x_response = tf.placeholder(tf.int32, shape=([None, hyper.max_timesteps]), name='x_response')
    y = tf.placeholder(tf.int32, shape=([None, const.num_classes]), name='y')

    # Formatting inputs
    x_c_emb = tf.nn.embedding_lookup(embeddings, x_context, name='x_c_emb')
    x_r_emb = tf.nn.embedding_lookup(embeddings, x_response, name='x_r_emb')
    y_onehot = tf.cast(tf.one_hot(y, int(const.num_classes), name='y_onehot'), tf.int64)
    print('context:', x_context.shape, '\tembedded:', x_c_emb.shape)
    print('response:', x_response.shape, '\tembedded:', x_r_emb.shape)
    print('labels:', y.shape)
    
    # Initialize network
    print('Initializing network architecture')
    rnn, initial_state = create_rnn(
            cell_type=hyper.cell_type,
            hidden_size=hyper.hidden_size,
            dropout_prob=hyper.dropout_prob,
            num_layers=hyper.num_layers,
            batch_size=hyper.batch_size)

    M= tf.get_variable('M', 
            shape=[hyper.batch_size, hyper.hidden_size, hyper.hidden_size], 
            initializer=hyper.initializer())

    b= tf.get_variable('b',
            shape=[hyper.batch_size],
            initializer=hyper.initializer())

    logits = create_dual_encoder(
            context=x_c_emb,
            response=x_r_emb,
            cell=rnn,
            state=initial_state,
            weights=M,
            bias=b,
            initializer=hyper.initializer)

    training, accuracy = create_performance(labels=y_onehot, 
            logits=logits, 
            optimizer=hyper.optimizer,
            lr=hyper.initial_lr, 
            decay_rate=hyper.decay_rate, 
            decay_steps=hyper.decay_steps, 
            global_step=current_epoch)

    print('Training model')
    train(hyper, const, 
            filenames_train='./data/mini_train.json',
            filenames_valid='./data/mini_valid.json',
            training_op=training, 
            accuracy_op=accuracy, 
            save_path=args.save_directory,
            saved=args.saved, 
            epochs=args.num_epochs,
            batch_size=hyper.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Dual RNN encoder model for next utterance classification.')
    parser.add_argument('save_directory', help='Where to save/load the training model')
    parser.add_argument('-p', '--param_line', type=int, default=-1, help='Line in params.txt')
    parser.add_argument('-n', '--num_epochs', type=int, default=1)
    parser.add_argument('-s', '--saved', 
            action='store_true', 
            default=False, 
            help='Whether to load a saved model')
    args = parser.parse_args()
    main(args)
