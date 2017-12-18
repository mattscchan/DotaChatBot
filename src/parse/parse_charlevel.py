import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET

def main(args):
	tree = ET.parse(args.filename)
	xml_root = tree.getroot()
	chars = []
	for convo in xml_root:
		for utterance in convo:
			if utterance.text == None:
				chars += list('UNK')
			else:
				chars += list(utterance.text)
		for c in chars:
			print(ord(c)-32)	
		break

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)