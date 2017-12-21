import tensorflow as tf
import argparse
import sys
import csv
import re

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _convert(context, target):
    return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'context': _int64_feature(context),
                    'target': _int64_feature(target)
        })).SerializeToString()

def _write_tfrecords(filename, examples_list):
    writer = tf.python_io.TFRecordWriter(filename)

    for example in examples_list:
        writer.write(example)
    writer.close()

def main(args):
    examples = []
  
    with open(args.filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        index = 0
        for row in reader:
            if int(row[2]) == 0:
                continue
            examples.append(_convert(int(row[0]), int(row[1])))
            if index % 1000000:
                print('We are at', index)
            index += 1

    print('Writing now...')
    _write_tfrecords('./data/100k_skipgrams.tfrecords', examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args)