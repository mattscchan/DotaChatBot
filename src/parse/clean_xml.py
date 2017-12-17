import argparse
import re

def main(args):
	clean_file = []
	with open(args.filename + '.xml', 'r') as f:
		for line in f:
			line = re.sub(r'&','&amp;', line)
			line = re.sub(r'<(?!/utt>|/s>|/data>|utt|data|s sid="\d*")', '&lt;', line)
			line = re.sub(r'[^a-zA-Z_\'\\"0-9?!<>,\.:;\[\]{}`~@#$%^*()+=|\- /]', '', line)
				
			clean_file.append(line)

	with open(args.filename + '_clean.xml', 'w+') as f2:
		for line in clean_file:
			f2.write(line)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)