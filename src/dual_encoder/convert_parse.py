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

def convert(in_data, out_path):
    print("Converting: " + out_path)

    def wrap_int64(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def wrap_bytes(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        for i, label in enumerate(in_data.label):
            context_bytes = in_data.context[i].tobytes()
            response_bytes = in_data.response[i].tobytes()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                    {
                        'context': wrap_bytes(context_bytes),
                        'response': wrap_bytes(response_bytes),
                        'label': wrap_int64(label)
                    }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

def parse(serialized):
    features = \
            {
                'context': tf.FixedLenFeature([], tf.string),
                'response': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }

    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    context = parsed_example['context']
    response = parsed_example['response']

    # Decode the raw bytes so it becomes a tensor with type.
    context = tf.decode_raw(context, tf.uint8)
    response = tf.decode_raw(response, tf.uint8)

    # The type is now uint8 but we need it to be float.
    context = tf.reshape(tf.cast(context, tf.float32), shape=[100])
    response= tf.reshape(tf.cast(response, tf.float32), shape=[100])

    # Get the label associated with the image.
    label = tf.reshape(tf.cast(parsed_example['label'], tf.float32), [-1])
    
    print(context.shape, response.shape, label.shape)

    # The image and label are now correct TensorFlow types.
    return context, response, label

