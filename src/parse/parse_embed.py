import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse
import re

def main(args):
	tree = ET.parse(args.filename)
	csv_file = re.sub(r'\.xml', '.csv', args.filename)
	xml_root = tree.getroot()
	new_arr = []

	count = 0
	for convo in xml_root:
		new_convo = []
		for utterance in convo:
			if utterance.text == None:
				new_convo.append('')
			else:
				new_convo.append(utterance.text)
		line = '\t'.join(new_convo)
		line += '\n'
		new_arr.append(line)

		if count % 100000 == 0:
			with open(csv_file, 'a', encoding='utf-8') as f:
				for el in new_arr:
					f.write(el)
		count += 1 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)