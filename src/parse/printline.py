import argparse

def main(args):
	with open(args.filename, 'r') as f:
		count = 1
		for line in f:
			if count == args.line:
				print(line)
				break
			count += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('line', type=int)
	args = parser.parse_args()
	main(args)