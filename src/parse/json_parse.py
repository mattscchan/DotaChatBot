import tensorflow as tf
import argparse
import numpy as np
import json
import collections
import random
import re
import os
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict

# Note that the char mappings are just ASCII value of char minus the lowest value (space) = 2
# 0 is the padding value, 1 is the UNK token and 2 is newline

# FOR SCIENCE
RAND_SEED = int(0xCAFEBABE)
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
word_vectors = KeyedVectors.load_word2vec_format('data/gensim_embeddings250.txt', binary=False)
word_dict = defaultdict(int, zip(word_vectors.index2word, [i for i in range(len(word_vectors.index2word))]))

def str_to_int(utt, token_type='str', newline=True):
        arr = []

        if token_type == 'str':
                for c in list(utt):
                        if c == 'π':
                                arr.append(1)
                        else:
                                arr.append(ord(c)-29)

                if newline:
                        arr.append(2)
        else:
                arr = [word_dict[t] for t in utt.split(' ')]

        return arr

class SequenceBuff:
	def __init__(self, buffersize):
		self.buffersize = buffersize
		self.capacity = 0
		self.internalMem = []
		
	def poll(self, token_type):
		if self.capacity > 2:
			index = random.randint(0, self.capacity-1)
		else:
			index = 0
		# make sure to shuffle the buffer
		if self.capacity < self.buffersize:
			np.random.shuffle(self.internalMem)
		return str_to_int(self.internalMem[index], token_type=token_type)

	def update(self, utterance):
		# replace

		if self.capacity > self.buffersize:
			index = random.randint(0, self.capacity-1)
			self.internalMem[index] = utterance
		# add
		else:
			self.internalMem.append(utterance)
			self.capacity += 1

def main(args):
	buff = SequenceBuff(args.buffersize)
	write_arr = []
	count = 0
	if args.type:
		tok = 'token_'
	else:
		tok = ''
	train_file = './data/' + tok + 'train.json'
	valid_file = './data/' + tok + 'valid.json'
	test_file = './data/' + tok + 'test.json'

	with open(args.filename, 'r', encoding='utf-8') as raw:

		for line in raw:
			obj = json.loads(line)
			chat = obj['chat']

			for utt in chat:
				buff.update(utt)

			context = []
			# at least enough room for context
			if len(chat) > (args.context + 1):
				# Choose random starting point if possible
				if len(chat)-args.context-1 > 0:
					start = random.randint(0, len(chat)-args.context-1)
				else:
					start = 0

				for index in range(start, start+args.context):
					context += str_to_int(chat[index], args.type)
				
				next_utt = str_to_int(chat[start + args.context], args.type)
			# need to shrink context
			else:
				if len(chat) > 2:
					for index in range(0, len(chat)-1):
						context += str_to_int(chat[index], args.type)
					next_utt = str_to_int(chat[len(chat)-1], args.type)
				else:
					continue

			fake_utt = buff.poll(args.type)
			
			obj_real = {'context': context, 'next_utt': next_utt, 'label': 0}
			obj_fake = {'context': context, 'next_utt': fake_utt, 'label': 1}
			
			write_arr.append(obj_real)
			write_arr.append(obj_fake)

			count += 1

			if count % 100000 == 0:
				print('We at', count)

				np.random.shuffle(write_arr)
				total = len(write_arr)
				train_size = int(total*0.8)
				valid_size = int((total - train_size)/2)
				test_size = int(total - train_size - valid_size)

				train_arr = write_arr[:train_size]
				valid_arr = write_arr[train_size:train_size+valid_size]
				test_arr = write_arr[train_size+valid_size:]

				# Train file
				with open(train_file, 'a', encoding='utf-8') as train:
					for el in train_arr:
						train.write(json.dumps(el))
						train.write('\n')
						
				# Valid file
				with open(valid_file, 'a', encoding='utf-8') as valid:
					for el in valid_arr:
						valid.write(json.dumps(el))
						valid.write('\n')

				# Test file
				with open(test_file, 'a', encoding='utf-8') as test:
					for el in test_arr:
						test.write(json.dumps(el))
						test.write('\n')

				del write_arr[:]
				del train_arr
				del valid_arr
				del test_arr

	print('Last write!', count)
	np.random.shuffle(write_arr)
	total = len(write_arr)
	train_size = int(total*0.8)
	valid_size = int((total - train_size)/2)
	test_size = int(total - train_size - valid_size)

	train_arr = write_arr[:train_size]
	valid_arr = write_arr[train_size:train_size+valid_size]
	test_arr = write_arr[train_size+valid_size:]

	# Train file
	with open(train_file, 'a', encoding='utf-8') as train:
		for el in train_arr:
			train.write(json.dumps(el))
			train.write('\n')
			
	# Valid file
	with open(valid_file, 'a', encoding='utf-8') as valid:
		for el in valid_arr:
			valid.write(json.dumps(el))
			valid.write('\n')

	# Test file
	with open(test_file, 'a', encoding='utf-8') as test:
		for el in test_arr:
			test.write(json.dumps(el))
			test.write('\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('buffersize', type=int)
	parser.add_argument('context', type=int)
	parser.add_argument('-t', '--type', action='store_true', default=False)
	args = parser.parse_args()
	main(args)
