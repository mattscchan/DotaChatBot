import tensorflow as tf
import argparse
import sys
import json


def _bytes_feature(value):
    value = ' '.join(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[[value.encode('utf-8')]]))

def _convert(text):
    return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'text': _bytes_feature(text),
        })).SerializeToString()

def _write_tfrecords(filename, examples_list):
    writer = tf.python_io.TFRecordWriter(filename)

    for example in examples_list:
        writer.write(example)
    writer.close()

def main(args):
    examples = []

    with open(args.filename, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            examples.append(_convert(obj['chat']))

    _write_tfrecords('./data/billion_chats.tfrecords', examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args)