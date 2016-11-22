
from gensim.models import word2vec
import time


def convert_corpora_to_sentence_iter():
    t8 = word2vec.Text8Corpus('text8/text8')
    return t8


def train_and_save_model(sentences, fname, **kwargs):
    w2v = word2vec.Word2Vec(sentences, **kwargs)
    w2v.save(fname)
    return w2v

def get_model_name(**kw):
    return u"w2v_model_size_{model_size}_window_{window}.model".format(**kw)

print __name__
if __name__ == '__main__':
    sentences_iter = convert_corpora_to_sentence_iter()
    for model_size in range(10, 200, 10):
        for window in (5, 7, 10):
            t0 = time.clock()
            print "Training with size={}, window={}".format(model_size, window)
            fname = get_model_name(model_size=model_size, window=window)
            model = train_and_save_model(sentences_iter, fname, size=model_size, window=window)
            t1 = time.clock()
            print "Training took {} secs".format(t1-t0)
