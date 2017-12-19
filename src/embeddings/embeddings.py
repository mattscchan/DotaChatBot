import tensorflow as tf
import argparse
import json
import re
from collections import Counter
import os.path as path
import numpy as np

VOCAB_SIZE = 500000

def main(args):
	word_counts = Counter()

	convo_num = 0
	total_utt = np.int64(0)
	total_tokens = np.int64(0)
	unique_tokens = np.int64(0)
	
	with open(args.filename, 'r') as f:
		for line in f:
			obj = json.loads(line)
			chat = obj['chat']

			for utt in chat:
				arr = utt.split()
				arr = [word for word in arr if arr != '']
				total_tokens += len(arr)
				word_counts.update(arr)
				total_utt =+ 1

			convo_num += 1

		dictionary = dict()
		unique_tokens = len(word_counts)
		top_n = word_counts.most_common(VOCAB_SIZE+1)
		

		# going to save dicts for a bunch of them since it takes so long to search the whole file
		# choosing 50k, 100k, 250k, 500k
		index = 0
		for word in top_n:
			dictionary[word[0]] = index
			index += 1
		
		rev_dict = dict(zip(dictionary.values(), dictionary.keys()))

		with open('50k_vocab.txt', 'w') as f2:
			for index in range(0, 50000):
				f2.write(rev_dict[index])
				f2.write('\n')

		with open('100k_vocab.txt', 'w') as f2:
			for index in range(0, 100000):
				f2.write(rev_dict[index])
				f2.write('\n')

		with open('250k_vocab.txt', 'w') as f2:
			for index in range(0, 250000):
				f2.write(rev_dict[index])
				f2.write('\n')

		with open('500k_vocab.txt', 'w') as f2:
			for index in range(0, 500000):
				f2.write(rev_dict[index])
				f2.write('\n')


		stats = []
		stats.append(convo_num)
		stats.append(unique_tokens)
		stats.append(total_tokens)
		stats.append(total_tokens)

		with open('corpus_chars.txt', 'w') as f2:
			for el in stats:
				print(el)
				f2.write(el)
				f2.write('\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)