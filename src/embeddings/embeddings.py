import tensorflow as tf
import argparse
import csv
from collections import Counter

def main(args):
	word_counts = Counter()
	csv.field_size_limit(2147483647)

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)