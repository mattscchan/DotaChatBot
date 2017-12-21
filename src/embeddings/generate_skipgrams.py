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

SEED = int('0xCAFEBABE', 16)
np.random.seed(SEED)
tf.set_random_seed(SEED)

def read_data(datafile):
	data = []
	with open(datafile, 'r', encoding='utf-8') as f:
		for line in f:
			obj = json.loads(line)
			chat = obj["chat"]
			data.append(chat)
	return data

def main(args):
	window_size = 3

	vocab_str = re.sub(r'\D', '', args.data)
	vocab_size = int(vocab_str) * 1000

	data = read_data(args.data)	
	sampling_table = sequence.make_sampling_table(vocab_size)

	word_target = []
	word_context = []
	labels = []
	print("DATA LOADED!")
	count = 0
	for convo in data:
		couples, tmp_labels = skipgrams(convo, vocab_size, window_size=window_size, sampling_table=sampling_table)
		if len(couples) < 1:
			continue
		labels += tmp_labels
		tmp_target, tmp_context = zip(*couples)
		word_target += tmp_target
		word_context += tmp_context

		count += 1
		
		if count % 1000000 == 0:
			print(count)
			with open('./data/'+vocab_str+'_skipgrams.txt', 'a', encoding='utf-8') as f:
				for index in range(0, len(word_context)):
					write_str = str(word_context[index]) + ','
					write_str += str(word_target[index])
					write_str += ','
					write_str += str(labels[index])
					write_str += '\n'
					f.write(write_str)
			del word_target[:]
			del word_context[:]
			del labels[:]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data')
	args = parser.parse_args()
	main(args)





