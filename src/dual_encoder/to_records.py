import convert_parse
import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def main():
    print('Loading data')
    data = [convert_parse.load_data('data/tiny_train.json', 100), 
            convert_parse.load_data('data/tiny_valid.json', 100), 
            convert_parse.load_data('data/tiny_test.json', 100)]

    filename = ['data/tiny_train.tfrecords',
            'data/tiny_valid.tfrecords',
            'data/tiny_test.tfrecords']
    for group in zip(data, filename):
        convert_parse.convert(*group)
    print('Conversion complete')

if __name__ == '__main__':
    main()
