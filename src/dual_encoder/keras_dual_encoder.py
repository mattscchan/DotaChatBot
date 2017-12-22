import re
import json
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras import optimizers, initializers
from gensim.models.keyedvectors import KeyedVectors

SEED = int('0xCAFEBABE', 16)
np.random.seed(SEED)
tf.set_random_seed(SEED)

# HYPERPARAMETERS
Hyper = namedtuple('Hyper', 
        ['hidden_units', 'lr', 'clipnorm', 'batch_size', 'optimizer', 'kernel_init', 'recurrent_init', 'dropout'])
Const = namedtuple('Const', ['embedding_size', 'max_timesteps', 'vocab_size'])

# DATASET
def load_json(file):
    df = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            df.append(json.loads(line))
    return df

def load_data(file, max_timesteps):
    Data = namedtuple('Data', ['context', 'response', 'label'])

    j = load_json(file)
    d = Data(pad_sequences([i['context'] for i in j], maxlen=max_timesteps, padding='pre'),
             pad_sequences([i['next_utt'] for i in j], maxlen=max_timesteps, padding='pre'),
             np.array([i['label'] for i in j]))
    return d 

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

def model(const, hyper, train, valid, test=None, epochs=1, saved_name=None, saved=False, weights=[]):
    if saved:
        print('Loading Dual Encoder')
        dual_encoder = load_model(saved_name)
    else:
        print('Dual LSTM Encoder')
        print('Embedding')
        encoder = Sequential(name='Encoder')
        encoder.add(Embedding(input_dim=const.vocab_size,
                            output_dim=const.embedding_size,
                            mask_zero=True,
                            trainable=True,
                            input_length=const.max_timesteps,
                            weights=[weights],
                            #name='Embedding'
                            ))
        print('Layers')
        if not hyper.kernel_init:
            hyper.kernel_init='glorot_uniform'
        if not hyper.recurrent_init:
            hyper.recurrent_init='orthogonal'
        ret_seq=True
        for i, units in enumerate(hyper.hidden_units):
            if i == len(hyper.hidden_units)-1:
                ret_seq = False
            encoder.add(LSTM(units, return_sequences=ret_seq, 
                kernel_initializer=hyper.kernel_init, recurrent_initializer=hyper.recurrent_init))
        
        context = Input(shape=(const.max_timesteps,), dtype='int32', name='Context')
        response = Input(shape=(const.max_timesteps,), dtype='int32', name='Response')
        
        context_encoder = encoder(context)
        response_encoder = encoder(response)
        
        combined = combine(context_encoder, response_encoder)
        if hyper.dropout:
            dropout = Dropout(hyper.dropout)(combined)
        else:
            dropout = combined
        similarity = Dense((1), activation = "sigmoid", name='Output') (dropout)
        
        dual_encoder = Model([context, response], similarity, name='Dual_Encoder')
        dual_encoder.layers[1].trainable=False
    dual_encoder.summary()
    
    optimizer = hyper.optimizer(lr=hyper.lr, clipnorm=hyper.clipnorm)
    dual_encoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = LossHistory()
    checkpoint = ModelCheckpoint(saved_name, monitor='val_acc', mode='auto', save_best_only=True, verbose=1)
    callback_list = [checkpoint, history]

    dual_encoder.fit([train.context, train.response], train.label, 
              batch_size=hyper.batch_size, 
              epochs=epochs, 
              validation_data=([valid.context, valid.response], valid.label),
              shuffle=True, 
              callbacks=callback_list,
              )

    if test:
        score = dual_encoder.evaluate([test.context, test.response], test.label, batch_size=hyper.batch_size)
        test_acc = score[1]
    else:
        test_acc = None

    return history.train_acc, history.valid_acc, test_acc

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
    # ------------------------------------------------------------------------------------------------------ Word Vector
    print('LOADING WVEC\n------------')
    if args.words:
        word_vectors = KeyedVectors.load_word2vec_format('data/gensim_embeddings250.txt', binary=False)
        word_path = 'token_'
    else:
        word_path = ''

    # ------------------------------------------------------------------------------------------------------- Parameters
    hyper = Hyper(
        hidden_units=[200],
        lr=0.0001,
        clipnorm=10,
        batch_size=512,
        dropout=0.5,
        optimizer=optimizers.Adam,
        kernel_init=initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=SEED),
        recurrent_init=initializers.Orthogonal(seed=SEED),
    )
    if args.words:
        const = Const(
            embedding_size=300,
            max_timesteps=100,
            vocab_size=len(word_vectors.index2word)+1,
        )
    else:
        const = Const(
            embedding_size=16,
            max_timesteps=100,
            vocab_size=100,
        )
    alphabet = ''
    print(hyper)
    print(const)

    if args.words:
        weights= np.zeros((const.vocab_size, const.embedding_size))
        for i, v in enumerate(word_vectors.syn0):
            weights[i, :] = np.array(v)
        weights = np.array(weights)
        print(weights.shape, type(weights), weights)
    else:
        weights = []

    # --------------------------------------------------------------------------------------------------------- Training
    print('LOADING DATA\n------------')
    train = load_data('data' +'/'+ args.mini_data + args.tiny_data + word_path + 'train.json', const.max_timesteps)
    valid = load_data('data' +'/'+ args.mini_data + args.tiny_data + word_path + 'valid.json', const.max_timesteps)
    test = load_data('data' +'/'+ args.mini_data + args.tiny_data + word_path + 'test.json', const.max_timesteps)
    print('Training data:', train.context.shape)
    print('Validation data:', valid.context.shape)
    print('Testing data:', test.context.shape)


    print('TRAINING MODEL\n--------------')
    saved_name = args.save_directory + '/' + args.mini_data + args.tiny_data + word_path + 'best.keras'
    log_name = args.save_directory + '/' + args.mini_data + args.tiny_data + word_path + 'best.log'
    train_acc, valid_acc, test_acc = model(const, 
            hyper, 
            train, 
            valid, 
            test, 
            epochs=args.num_epochs, 
            saved_name=saved_name, 
            saved=args.saved,
            weights=weights,
            )
    log_history(train_acc=str(train_acc), valid_acc=str(valid_acc),test_acc=str(test_acc), path=log_name)

if __name__ == '__main__':
    # TODO load model, automatic naming of model
    parser = argparse.ArgumentParser(
            description='Dual LSTM encoder model for next utterance classification.')
    parser.add_argument('save_directory', help='Where to save/load the training model')
    parser.add_argument('-m', '--mini_data', action='store_const', const= 'mini_', default='')
    parser.add_argument('-t', '--tiny_data', action='store_const', const= 'tiny_', default='')
    parser.add_argument('-s', '--saved', action='store_true', default=False)
    parser.add_argument('-n', '--num_epochs', type=int, default=1)
    parser.add_argument('-w', '--words', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
