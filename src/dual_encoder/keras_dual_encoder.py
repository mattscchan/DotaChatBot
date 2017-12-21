import convert_parse

import os
import re
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras import optimizers, initializers

from tensorflow.python.keras.estimator import model_to_estimator

SEED = int('0xCAFEBABE', 16)
np.random.seed(SEED)
tf.set_random_seed(SEED)

# HYPERPARAMETERS
Hyper = namedtuple('Hyper', ['hidden_units', 'lr', 'clipnorm', 'batch_size', 
    'optimizer', 'kernel_init', 'recurrent_init', 'dropout'])
Const = namedtuple('Const', ['embedding_size', 'max_timesteps', 'vocab_size', 'total_n'])

def input_fn(filenames, shuffle, batch_size=32, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(convert_parse.parse)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None # Allow infinite reading of the data.
    else:
        num_repeat = 1

    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    print(dataset.output_shapes)

    context_batch, response_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    #x = {'context': context_batch, 'response': response_batch}
    #y = labels_batch

    return context_batch, response_batch, labels_batch

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.valid_acc = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.valid_acc.append(logs.get('val_acc'))

def combine(c, r):
    return concatenate([c, r], name='Combine')

def model(const, hyper, train, valid, test=None, epochs=1, saved_name=None, saved=False):

    # Get data from Dataset API
    train_c, train_r, train_l = input_fn(train, shuffle=True, batch_size=hyper.batch_size)
    valid_c, valid_r, valid_l = input_fn(train, shuffle=False, batch_size=hyper.batch_size)
    #test_d = input_fn(train, shuffle=False, batch_size=hyper.batch_size)
    print(train_c.shape)

    if saved:
        print('Loading Dual Encoder')
        dual_encoder = load_model(saved_name)
    else:
        print('Dual LSTM Encoder')
        encoder = Sequential(name='Encoder')
        encoder.add(Embedding(input_dim=const.vocab_size,
                            output_dim=const.embedding_size,
                            mask_zero=True,
                            trainable=True,
                            input_length=const.max_timesteps,
                            #weights=[],
                            name='Embedding'
                            ))
        if not hyper.kernel_init:
            hyper.kernel_init='glorot_uniform'
        if not hyper.recurrent_init:
            hyper.recurrent_init='orthogonal'

        ret_seq=True
        for i, units in enumerate(hyper.hidden_units):
            if i == len(hyper.hidden_units)-1: ret_seq = False
            encoder.add(LSTM(units, 
                return_sequences=ret_seq, 
                kernel_initializer=hyper.kernel_init, 
                recurrent_initializer=hyper.recurrent_init))
        
        #context = Input(shape=(const.max_timesteps,), dtype='int32', name='context')
        #response = Input(shape=(const.max_timesteps,), dtype='int32', name='response')
        context = Input(tensor=train_c, name='context')
        response = Input(tensor=train_r, name='response')
        label = Input(tensor=train_l, name='label')
        
        context_encoder = encoder(context)
        response_encoder = encoder(response)
        
        combined = combine(context_encoder, response_encoder)
        if hyper.dropout:
            combined = Dropout(hyper.dropout)(combined)
        similarity = Dense((1), activation = "sigmoid", name='Output') (combined)
        
        dual_encoder = Model([context, response], similarity, name='Dual_Encoder')
    
    dual_encoder.summary()
    optimizer = hyper.optimizer(lr=hyper.lr, clipnorm=hyper.clipnorm)

    dual_encoder.compile(optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=['accuracy'], 
            target_tensors=[label],
            )

    dual_encoder.fit(#None, label, 
            #batch_size=hyper.batch_size,
            steps_per_epoch=int(const.total_n/hyper.batch_size), 
            epochs=epochs, 
            #validation_data=([valid_c,valid_r], valid_l),
            #validation_steps=10,
            )

    # Covert Keras to Estimator
    #de_estimator = model_to_estimator(keras_model=dual_encoder, model_dir=saved_name)
    #de_estimator.train(input_fn=lambda:input_fn(train, shuffle=True, batch_size=hyper.batch_size))
    #train_op = tf.estimator.TrainSpec(input_fn=lambda:input_fn(train, shuffle=True, batch_size=hyper.batch_size))
    #valid_op = tf.estimator.EvalSpec(input_fn=lambda:input_fn(valid, shuffle=True, batch_size=hyper.batch_size))
    #tf.estimator.train_and_evaluate(de_estimator, train_op, valid_op)

    #test_op = de_estimator.predict(input_fn=lambda:input_fn(test, labels=None

    #return history.train_acc, history.valid_acc, test_acc

def clean_funcname(string):
    if string:
        string = re.sub("<class '", "", string)
        string = re.sub("'>", "", string)
        string = re.sub("tensorflow.python.keras._impl.keras.", "", string)
    else:
        string = ''
    return string

def log_history(train_acc, valid_acc, path, test_acc=None):
    train_acc = clean_funcname(train_acc)
    valid_acc = clean_funcname(valid_acc)
    test_acc = clean_funcname(test_acc)
    with open(path, 'a', encoding='utf-8') as path:
        path.write(train_acc + ',' + valid_acc + ',' + test_acc + '\n')

def main(args):
    # Parameters
    hyper = Hyper(
        hidden_units=[100],
        lr=0.0001,
        clipnorm=0,
        batch_size=256,
        dropout=None,
        optimizer=optimizers.Adam,
        kernel_init=initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=SEED),
        recurrent_init=initializers.Orthogonal(seed=SEED),
    )
    const = Const(
        embedding_size=16,
        max_timesteps=100,
        vocab_size=100,
        total_n=13071798,
    )

    print('TRAINING MODEL\n--------------')
    saved_name = args.save_directory + '/' + args.mini_data + args.tiny_data + 'best.keras'
    saved_name = os.path.join(os.getcwd(), saved_name)
    print(saved_name)
    model(const, 
            hyper, 
            train='data/' + args.mini_data + args.tiny_data + 'train.tfrecords', 
            valid='data/' + args.mini_data + args.tiny_data + 'valid.tfrecords', 
            test='data/' + args.mini_data + args.tiny_data + 'test.tfrecords', 
            epochs=args.num_epochs, 
            saved_name=saved_name)

if __name__ == '__main__':
    # TODO load model, automatic naming of model
    parser = argparse.ArgumentParser(
            description='Dual LSTM encoder model for next utterance classification.')
    parser.add_argument('save_directory', help='Where to save/load the training model')
    parser.add_argument('-m', '--mini_data', action='store_const', const= 'mini_', default='')
    parser.add_argument('-t', '--tiny_data', action='store_const', const= 'tiny_', default='')
    parser.add_argument('-s', '--saved', action='store_true', default=False)
    parser.add_argument('-n', '--num_epochs', type=int, default=1)
    args = parser.parse_args()
    main(args)
