import argparse
import json
import re

def read_data(vectorfile):
	dictionary = {}

	with open(vectorfile, 'r', encoding='utf-8') as f:
		index = 0
		for line in f:
			line = re.sub(r'\n', '', line)
			dictionary[line] = index
			index += 1
	rev_dict = dict(zip(dictionary.value(), dictionary.keys()))

	return rev_dict

def convert_to_indices(datafile, rev_dict, out):
	write_arr = []
	count = 0
	with open(datafile, 'r', encoding='utf-8') as f:
		for line in f:
			obj = json.loads(line)
			chat = obj["chat"]
			chat = [utt.split() for utt in chat]
			chat = [rev_dict[word] if word in rev_dict else 0 for word in chat]

			new_obj = {"chat": chat}
			write_arr.append(new_obj)

			count += 1

			if count%100000 == 0:
				print("Count as", count)
				with open(output, 'a', encoding='utf-8') as f2:
					for el in write_arr:
						f2.write(json.dumps(el))
						f2.write('\n')
			write_arr[:]

	with open(data.output, 'a', encoding='utf-8') as f2:
		print("FINAL PRINT", count)
		for el in write_arr:
			f2.write(json.dumps(el))
			f2.write('\n')


def main(args):
	for i in range(0, 4):
		if i==0:
			vectors = '50k_vocab.txt'
			out = '50k_billions.json'
		elif i==1:
			vectors = '100k_vocab.txt'
			out = '100k_billions.json'
		elif i==2:
			vectors = '250k_vocab.txt'
			out = '250k_billions.json'
		else:
			vectors = '500k_vocab.txt'
			out = '500k_billions.json'

		rev_dict = read_data(vectors)
		convert_to_indices(args.data, rev_dict, out)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data')
	parseer.add_argument('output')
	main()