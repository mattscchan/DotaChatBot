from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.layers.merge import concatenate
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

def read_vectors(vectorfile):
	dictionary = {}
	index = 0
	with open(vectorfile, 'r', encoding='utf-8') as f:
		for line in f:
			line = re.sub('\n', '', line)
			dictionary[index] = line
			index += 1
	return dictionary

def read_data(datafile):
	word_context = []
	word_target = []
	labels = []
	with open(datafile, 'r', encoding='utf-8') as f:
		for line in f:
			row = line.split(',')
			word_context.append(int(row[0]))
			word_target.append(int(row[1]))
			labels.append(int(row[2]))
	return word_context, word_target, labels

def main(args):
	window_size = 3
	vector_dim = 300
	epochs = 2000000

	valid_size = 16     # Random set of words to evaluate similarity on.
	valid_window = 100  # Only pick dev samples in the head of the distribution.
	valid_examples = np.random.choice(valid_window, valid_size, replace=False)

	vocab_str = re.sub(r'\D', '', args.data)
	vocab_size = int(vocab_str) * 1000

	reverse_dictionary = read_vectors('./data/100k_vocab.txt')
	print('Dict done!')
	word_context, word_target, labels = read_data(args.data)	
	print("Read all data!")

	word_target = np.array(word_target, dtype="int32")
	word_context = np.array(word_context, dtype="int32")

	input_target = Input((1,))
	input_context = Input((1,))

	embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
	target = embedding(input_target)
	target = Reshape((vector_dim, 1))(target)
	context = embedding(input_context)
	context = Reshape((vector_dim, 1))(context)

	similarity = concatenate([target, context], mode='cos', dot_axes=0)

	# now perform the dot product operation to get a similarity measure
	dot_product = concatenate([target, context], mode='dot', dot_axes=1)
	dot_product = Reshape((1,))(dot_product)
	# add the sigmoid output layer
	output = Dense(1, activation='sigmoid')(dot_product)
	# create the primary training model
	model = Model(input=[input_target, input_context], output=output)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')

	validation_model = Model(input=[input_target, input_context], output=similarity)

	class SimilarityCallback:
	    def run_sim(self):
	        for i in range(valid_size):
	            valid_word = reverse_dictionary[valid_examples[i]]
	            top_k = 8  # number of nearest neighbors
	            sim = self._get_sim(valid_examples[i])
	            nearest = (-sim).argsort()[1:top_k + 1]
	            log_str = 'Nearest to %s:' % valid_word
	            for k in range(top_k):
	                close_word = reverse_dictionary[nearest[k]]
	                log_str = '%s %s,' % (log_str, close_word)
	            print(log_str)

	    @staticmethod
	    def _get_sim(valid_word_idx):
	        sim = np.zeros((vocab_size,))
	        in_arr1 = np.zeros((1,))
	        in_arr2 = np.zeros((1,))
	        in_arr1[0,] = valid_word_idx
	        for i in range(vocab_size):
	            in_arr2[0,] = i
	            out = validation_model.predict_on_batch([in_arr1, in_arr2])
	            sim[i] = out
	        return sim
	sim_cb = SimilarityCallback()

	arr_1 = np.zeros((1,))
	arr_2 = np.zeros((1,))
	arr_3 = np.zeros((1,))

	for cnt in range(epochs):
	    idx = np.random.randint(0, len(labels)-1)
	    arr_1[0,] = word_target[idx]
	    arr_2[0,] = word_context[idx]
	    arr_3[0,] = labels[idx]
	    loss = model.train_on_batch([arr_1, arr_2], arr_3)
	    if cnt % 10000 == 0:
	        print("Iteration {}, loss={}".format(cnt, loss))
	    if cnt % 50000 == 0:
        	sim_cb.run_sim()
        	model.save_weights('./data/models/embeddings_'+str(cnt)+'.h5')
        	mode.save('./data/models/graph_'+str(cnt)+'.h5')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data')
	args = parser.parse_args()
	main(args)