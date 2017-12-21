import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    print('Loading data')
    data = [load_data('data/tiny_train.json', 100), 
            load_data('data/tiny_valid.json', 100), 
            load_data('data/tiny_test.json', 100)]

    filename = ['tiny_train',
                'tiny_valid',
                'tiny_test']
    length = [len(i.context) for i in data]
    length.append(0)
    total = sum(length)

    writer = [tf.python_io.TFRecordWriter(f + '.records') for f in filename]

    print('Writing TFRecords')
    for j, d in enumerate(data):
        for i, label in enumerate(d.label):
            feature  = {filename[j]+'/cr':_bytes_feature(np.concatenate((d.context[i], d.response[i])).tobytes()),
                        filename[j]+'/y':_int64_feature(label)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer[j].write(example.SerializeToString())
            if i % 1000 == 0:
                print('Processing: %8d/%d' % (i + length[j-1], total), end='\r', flush=True)
        writer[j].close()
    print('Processing: %8d/%d' % (i + length[j-1], total))

if __name__ == '__main__':
    main()
