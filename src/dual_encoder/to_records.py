import json
import numpy as np
import tensorflow as tf
from collections import namedtupple
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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
    d = Data(pad_sequences([i['context'] for i in j], maxlen=max_timesteps),
             pad_sequences([i['next_utt'] for i in j], maxlen=max_timesteps),
             np.array([i['label'] for i in j]))
    return d 

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main():
    data = [load_data('data/train.json', 100), 
            load_data('data/valid.json', 100), 
            load_data('data/test.json', 100)]

    filename = ['train',
                'valid',
                'test']

    writer = [tf.python_io.TFRecordWriter(f + '.records') for f in filename]

    for j, d in enumerate(data):
        for i, label in enumerate(d.label):
            feature  = {filename[j]+'/cr':_int64_feature(d.context[i].extend(d.response[i])),
                        filename[j]+'/y':_int64_feature(label)}
            example = tf.train.Example(features=tf.train.features(feature=feature))
            writer[j].write(example.SerializeToTring())
        writer[j].close()
