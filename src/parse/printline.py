import argparse
import json

def main(args):
	with open(args.filename, 'r') as f:
		count = 1
		for line in f:
			if count == args.line:
				print(line)
				if args.json:
					obj = json.loads(line)
					print(obj['context_size'])
					print(obj['context'])
					print(obj['next_utt'])
					print(obj['label'])
				break
			count += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('line', type=int)
	parser.add_argument('-json', '--json', type=bool)
	args = parser.parse_args()
	main(args)