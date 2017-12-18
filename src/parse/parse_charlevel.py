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
				chars.append(list('UNK'))
			else:
				chars.append(list(utterance.text))
		for c in chars:
			print(type(c))
			print(ord(c)-32)	
		break

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)