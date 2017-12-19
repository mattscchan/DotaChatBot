import tensorflow as tf
import xml.etree.ElementTree as ET
import argparse
import re
# Note that the char mappings are just ASCII value of char minus the lowest value (space) = 2
# 0 is the padding value, 1 is the UNK token and 2 is newline

# FOR SCIENCE
RAND_SEED = int(0xCAFEBABE)
random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

def str_to_int(utt, newline=True):
	arr = []

	for c in list(utt):
		if c == 'π':
			arr.append(1)
		else:
			arr.append(ord(c)-27)

	if newline:
		arr.append(2)
		return arr
	return arr


class SequenceBuff:

	def __init__(self, buffersize):
		self.buffersize = buffersize
		self.capacity = 0
		self.internalMem = []
		
	def poll(self):
		if self.capacity > 2:
			index = random.randint(0, self.capacity-1)
		else:
			index = 0
		# make sure to shuffle the buffer
		if self.capacity < self.buffersize:
			np.random.shuffle(self.internalMem)
		return str_to_int(self.internalMem[index])

	def update(self, utterance):
		# replace

		if self.capacity > self.buffersize:
			index = random.randint(0, self.capacity-1)
			self.internalMem[index] = utterance
		# add
		else:
			self.internalMem.append(utterance)
			self.capacity += 1

def main(args):
	tree = ET.iterparse(args.filename)
	json_file = re.sub(r'\.xml', '.json', args.filename)

	buff = SequenceBuff(args.buffersize)
	json_objs = []
	convo_num = 0

	for event, elem in tree:

		if event == 'end':
			if elem.tag == 's':
				utt_num = 0
				convo = []
				for utterance in elem:
					if not re.search(r'\n|\t', utterance.text):
						convo.append(utterance.text)
						buff.update(utterance.text)

					utterance.clear()
					
				obj_real = {}
				obj_fake = {}
				context = []

				if utt_num < 2:
					continue
				elif utt_num-1 < args.context:
					obj_real['context_size'] = utt_num-1
					obj_fake['context_size'] = utt_num-1
					for i in range(0, utt_num-1):
						context += str_to_int(convo[i])
						
					next_utt = str_to_int(convo[utt_num-1])
				else:
					obj_real['context_size'] = args.context
					obj_fake['context_size'] = args.context
					for i in range(0, args.context):
						context += str_to_int(convo[i])
					next_utt = str_to_int(convo[args.context])
				fake_utt = buff.poll()

				obj_real['next_utt'] = next_utt
				obj_real['context'] = context
				obj_real['label'] = 0

				obj_fake['context'] = context
				obj_fake['next_utt'] = fake_utt
				obj_fake['label'] = 1 
				

				json_objs.append(obj_real)
				json_objs.append(obj_fake)

				convo_num += 1
				
				if convo_num % 10000 == 0:
					print(convo_num)
					with open(json_file, 'a', encoding='utf-8') as f:
						for obj in json_objs:
							f.write(json.dumps(obj))
							f.write('\n')
						del json_objs[:]


		


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('buffersize', type=int)
	parser.add('context', type=int)
	args = parser.parse_args()
	main(args)