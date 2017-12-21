import gensim, logging, argparse, json
from gensim.models.keyedvectors import KeyedVectors

SEED = int(0xCAFEBABE)

def read_data(filename):
	sentences = []
	with open(filename, 'r', encoding='utf-8') as f:
		for line in f:
			obj = json.loads(line)
			chat = obj['chat']
			tmp = []
			for utt in chat:
				tmp += utt.lower().split()
			sentences.append(tmp)
	return sentences

def main(args):
	vocab_size = 100000
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	examples = read_data(args.filename)

	class myText(object):
		def __iter__(self):
			for line in examples:
				yield line

	sentences = myText()
	model = gensim.models.Word2Vec(sentences, size=300, max_vocab_size=vocab_size, sg=1, sample=0.0001, seed=SEED, min_count=10, workers=16)

	print(model.wv.similar_by_word('gg', restrict_vocab=vocab_size))
	print(model.wv.similar_by_word('glhf', restrict_vocab=vocab_size))
	print(mode.wv.similar_by_word('gank', restrict_vocab=vocab_size))
	model.wv.save_word2vec_format('./data/gensim_embeddings.txt', binary=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	main(args)