import tensorflow as tf
import argparse
import numpy as np
import json
import collections
import random
import re
import os

# Note that the char mappings are just ASCII value of char minus the lowest value (space) = 2
# 0 is the padding value, 1 is the UNK token and 2 is newline

# FOR SCIENCE
RAND_SEED = int(0xCAFEBABE)
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

def str_to_int(utt, newline=True):
	arr = []

	for c in list(utt):
		if c == 'π':
			arr.append(1)
		else:
			arr.append(ord(c)-27)

	if newline:
		arr.append(2)
		return arr
	return arr


class SequenceBuff:
	def __init__(self, buffersize):
		self.buffersize = buffersize
		self.capacity = 0
		self.internalMem = []
		
	def poll(self):
		if self.capacity > 2:
			index = random.randint(0, self.capacity-1)
		else:
			index = 0
		# make sure to shuffle the buffer
		if self.capacity < self.buffersize:
			np.random.shuffle(self.internalMem)
		return str_to_int(self.internalMem[index])

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

	with open(args.filename, 'r', encoding='utf-8') as raw:
		for line in raw:
			obj = json.dumps(line)
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

				for index in range(start, args.context):
					context += str_to_int(chat[index])

				next_utt = str_to_int(chat[start + args.context])
			# need to shrink context
			else:
				for index in range(0, len(chat)-1):
					context += str_to_int(chat[index])
				next_utt = str_to_int(chat[len(chat)-1])

			fake_utt = buff.poll()

			obj_real = {'context': context, 'next_utt': next_utt, 'label': 0}
			obj_fake = {'context': context, 'next_utt': fake_utt, 'label': 1}
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('buffersize', type=int)
	parser.add_argument('context', type=int)
	args = parser.parse_args()
	main(args)