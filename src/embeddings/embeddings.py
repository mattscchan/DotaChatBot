import tensorflow as tf
import argparse
import csv
from collections import Counter
import os.path as path

def main(args):
	word_counts = Counter()
	csv.field_size_limit(2147483647)
	dictionary = {}
	
	with open(args.filename, newline='', encoding='utf-8') as f:
		csv_reader = csv.reader(f, delimiter='\t')
		# Each line is an example
		for line in csv_reader:
			for utterance in line:
				words_list = utterance.split(' ')
				words_list = [word.lower() for word in words_list]
				# Track word frequencies
				word_counts.update(words_list)
		print(len(word_counts))
		
		freq = []
		unk_count = 0
		for el in word_counts:
			if word_counts[el] < 2:
				unk_count += 1
			else:
				dictionary[el] = len(dictionary)
		word_counts['unk'] += unk_count
		print('UNK: ', word_counts['unk'])
		print(len(freq))

		index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

		with open(path.join(path.dirname(args.filename), 'vectors.txt'), encoding='utf-8') as f2:
			for index in range(0, len(index)):
				f2.write(index_dictionary[index])
				f2.write('\n')
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)