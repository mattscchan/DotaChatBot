from keras.models import Model
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import argparse
import json
import tensorflow as tf
import numpy as np

SEED = int('0xCAFEBABE', 16)
np.random.seed(SEED)
tf.set_random_seed(SEED)

def read_data(datafile, vectorfile):
	data = []
	dictionary = {}
	with open(datafile, 'r', encoding='utf-8') as f:
		for line in f:
			obj = json.loads(line)
			chat = obj["chat"]
			chat = [utt.split() for utt in chat]
			data.append(chat)

	with open(vectorfile, 'r', encoding='utf-8') as f:
		index = 0
		for line in f:
			line = re.sub(r'\n', '', line)
			dictionary[line] = index
			index += 1
	rev_dict = dict(zip(dictionary.value(), dictionary.keys()))

	return data, rev_dict

def convert_to_indices(data, rev_dict):
	converted = []
	for convo in data:
		convo = [word[]]


def main(args):
	vocab_size = 
	data, rev_dict = read_data(args.data, args.vectors)
	indexed_data = convert_to_indices(data, rev_dict)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vectors')
	parser.add_argument('data')
	main()