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
	
	with open(args.filename, 'r', encoding='utf-8') as f:
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
			
			if convo_num % 100000 == 0:
				print('We at', convo_num)
		unique_tokens = len(word_counts)

		

		stats = []
		stats.append(convo_num)
		stats.append(unique_tokens)
		stats.append(total_tokens)
		stats.append(total_tokens)

		with open('./data/corpus_chars.txt', 'w', encoding='utf-8') as f2:
			for el in stats:
				print(el)
				f2.write(el)
				f2.write('\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)