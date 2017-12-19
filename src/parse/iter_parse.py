import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse
import re

def main(args):
	tree = ET.iterparse(args.filename)
	
	for event, elem in tree:
		if event == 'end':
			if elem.tag == 's':
				arr = []
				for utt in elem:
					if not re.search(r'\n|\t', utt.text):
						arr.append(utt.text)
					utt.clear()
					
				print(arr)
				del arr

		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)