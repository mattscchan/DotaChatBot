import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET
import numpy as np

# Note that the char mappings are just ASCII value of char minus the lowest value (space) = 2
# 0 is the padding value and 1 is the UNK token


def main(args):
	tree = ET.parse(args.filename)
	xml_root = tree.getroot()
	chars = []
	max = -1
	utt_num = []
	for convo in xml_root:
		for utterance in convo:
			if utterance.text == None:
				chars += list('UNK')
			else:
				if max < len(utterance.text):
					max = len(utterance.text)
				chars += list(utterance.text)
				utt_num.append(len(utterance.text))
	print(np.mean(utt_num))
		# for c in chars:
		# 	print(ord(c)-30)	
		# break

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)