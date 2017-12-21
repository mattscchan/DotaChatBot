from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import argparse
import json
import tensorflow as tf
import numpy as np
import re
import csv

SEED = int('0xCAFEBABE', 16)
np.random.seed(SEED)
tf.set_random_seed(SEED)

def read_data(datafile):
	word_context = []
	word_target = []
	labels = []
	with open(datafile, 'r', encoding='utf-8') as f:
		file = csv.reader(f)
		for row in file:
			word_context.append(row[0])
			word_target.append(row[1])
			labels.append(row[2])
	return word_context, word_target, labels

def main(args):
	window_size = 3
	vector_dim = 300
	epochs = 200000

	vocab_str = re.sub(r'\D', '', args.data)
	vocab_size = int(vocab_str) * 1000

	word_context, word_target, labels = read_data(args.data)	
	# sampling_table = sequence.make_sampling_table(vocab_size)

	# word_target = []
	# word_context = []
	# labels = []
	# print("DATA LOADED!")
	# count = 0
	# for convo in data:
	# 	couples, tmp_labels = skipgrams(convo, vocab_size, window_size=window_size, sampling_table=sampling_table)
	# 	if len(couples) < 1:
	# 		continue
	# 	labels += tmp_labels
	# 	tmp_target, tmp_context = zip(*couples)
	# 	word_target += tmp_target
	# 	word_context += tmp_context


	# print("FINISHED GENERATING SKIP GRAMS!")
	# del data

	word_target = np.array(word_target, dtype="int32")
	word_context = np.array(word_context, dtype="int32")

	input_target = Input((1,))
	input_context = Input((1,))

	embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
	target = embedding(input_target)
	target = Reshape((vector_dim, 1))(target)
	context = embedding(input_context)
	context = Reshape((vector_dim, 1))(context)

	# now perform the dot product operation to get a similarity measure
	dot_product = merge([target, context], mode='dot', dot_axes=1)
	dot_product = Reshape((1,))(dot_product)
	# add the sigmoid output layer
	output = Dense(1, activation='sigmoid')(dot_product)
	# create the primary training model
	model = Model(input=[input_target, input_context], output=output)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')

	arr_1 = np.zeros((1,))
	arr_2 = np.zeros((1,))
	arr_3 = np.zeros((1,))

	for cnt in range(epochs):
	    idx = np.random.randint(0, len(labels)-1)
	    arr_1[0,] = word_target[idx]
	    arr_2[0,] = word_context[idx]
	    arr_3[0,] = labels[idx]
	    loss = model.train_on_batch([arr_1, arr_2], arr_3)
	    if cnt % 100 == 0:
	        print("Iteration {}, loss={}".format(cnt, loss))
	        embed = model.get_layer(name='embedding')
	        print(embed)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data')
	args = parser.parse_args()
	main(args)