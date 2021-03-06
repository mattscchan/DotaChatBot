import gensim, logging, argparse, json
from gensim.models.keyedvectors import KeyedVectors

def main(args):
	model = KeyedVectors.load_word2vec_format(args.vectors, binary=False)

	if args.target == '1':
		with open('./data/lookup_sim.txt', 'r', encoding='utf-8') as f:
			for line in f:
				print(model.similar_by_word(line))
	else:
		print(model.similar_by_word(args.target, topn=20))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('vectors')
	parser.add_argument('target')
	args = parser.parse_args()
	main(args)