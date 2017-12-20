import tensorflow as tf
import argparse
import random
import re
import numpy as np

RAND_SEED = int(0xCAFEBABE)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)

def load_table(vectorfile):
	values = []
	with open(vectorfile, 'r') as f:
		index = 0
		for line in f:
			line = re.sub(r'\n', '', line)
			# line = re.sub(r'1', 'one', line)
			# line = re.sub(r'2', 'two', line)
			# line = re.sub(r'3', 'three', line)
			# line = re.sub(r'4', 'four', line)
			# line = re.sub(r'5', 'five', line)
			# line = re.sub(r'6', 'six', line)
			# line = re.sub(r'7', 'seven', line)
			# line = re.sub(r'8', 'eight', line)
			# line = re.sub(r'9', 'nine', line)
			# line = re.sub(r'0', 'zero', line)
			values.append(line)
	return values

# def convert_words_to_index(words, dictionary):
#     """ Replace each word in the dataset with its index in the dictionary """
#     return [dictionary[word] if word in dictionary else 0 for word in words]

# def generate_sample(index_words, context_window_size):
# 	""" Form training pairs according to the skip-gram model. """
# 	for index, center in enumerate(index_words):
# 		context = random.randint(1, context_window_size)
# 		# get a random target before the center word
# 		for target in index_words[max(0, index - context): index]:
# 			yield center, target
# 		# get a random target after the center wrod
# 		for target in index_words[index + 1: index + context + 1]:
# 			yield center, target

# def get_batch(iterator, batch_size):
#     """ Group a numerical stream into batches and yield them as Numpy arrays. """
#     while True:
#         center_batch = np.zeros(batch_size, dtype=np.int32)
#         target_batch = np.zeros([batch_size, 1])
#         for index in range(batch_size):
#             center_batch[index], target_batch[index] = next(iterator)
#         yield center_batch, target_batch

def parse_JSON(example):
    feature = {
                "text": tf.FixedLenFeature([], tf.string)
            }
    obj_ex = tf.parse_example(example, feature)
    
    return tf.sparse_tensor_to_dense(tf.string_split(obj_ex["text"]), default_value='π')


def create_dataset(filenames, parse_function, table, context, num_parallel_calls=1, batch_size=32,  shuffle_buffer=10000, num_epochs=1):

    def generate_example(chat):
        target = random.randint(0, len(chat))   
        context = random.randint(max(target-context, 0), min(target+context, len(chat)-1))
        return chat[target], chat[context]

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.batch(32)
    dataset = dataset.map(parse_function, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(lambda x: table.lookup(x), num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(lambda x: tuple(tf.py_func(generate_example, [x], [tf.int64, tf.int64])), num_parallel_calls=num_parallel_calls)

    if num_epochs < 0:
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.prefetch(10000)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    return next_element, iterator


def main(args):
	values = load_table(args.vectors)
	table = tf.contrib.lookup.index_table_from_tensor(values, default_value=0, name="TABLE")
	
	filenames = tf.placeholder(tf.string, shape=[None])

	next_element, iterator = create_dataset(filenames, parse_JSON, table, args.context, num_parallel_calls=4)

	with tf.Session() as sess:
		sess.run(iterator.initializer, feed_dict={filenames: [args.data]})
		sess.run(tf.tables_initializer())
		sess.run(tf.global_variables_initializer())

		el = sess.run(next_element)
		print(el)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vectors')
	parser.add_argument('data')
	parser.add_argument('context', type=int)
	args = parser.parse_args()
	main(args)