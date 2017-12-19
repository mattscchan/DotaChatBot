import argparse
import re

def main(args):
	clean_file = []
	with open(args.filename+'_clean.xml', 'a') as f2:
		f2.write('<data>')

	with open(args.filename + '.xml', 'r') as f:
		count = 0
		for line in f:
			line = re.sub(r'&','&amp;', line)
			line = re.sub(r'<(?!/utt>|/s>|/data>|utt|data|s sid="\d*")', '&lt;', line)
			line = re.sub(r'[^a-zA-Z_\'\\"0-9?!<>,\.:;\[\]{}`~@#$%^*()+=|\- /\t\n\r]+', 'π', line)
				
			clean_file.append(line)
			count += 1
			if count % 1000000 == 0:
				with open(args.filename + '_clean.xml', 'a') as f2:
					for line in clean_file:
						f2.write(line)
				del clean_file[:]
				
		with open(args.filename + '_clean.xml', 'a') as f2:
			for line in clean_file:
				f2.write(line)
			f2.write('</data>')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)