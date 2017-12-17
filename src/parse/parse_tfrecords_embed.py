import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse

def main(args):
	tree = ET.parse(args.filename)
	xml_root = tree.getroot()
	
	count = 0
	for convo in xml_root:
		for utterance in convo:
			print(utterance.text)
			if count > 10:
				return
			count += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)