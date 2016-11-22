from main import get_model_name
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_completeness_v_measure as hom_v_score

NCLUSTERS = 50
results = {}
for window_size in (5,7,10):
    results[window_size] = []
    ref_model_name = get_model_name(model_size=190, window=window_size)
    ref_model = word2vec.Word2Vec.load(u'C:\\Users\\michar\\PycharmProjects\\play_gensim\\'+ref_model_name)
    ref_kmeans = KMeans(n_clusters=NCLUSTERS, random_state=0).fit(ref_model.syn0)
    for model_size in range(10, 200, 10):
        model_name = get_model_name(model_size=model_size, window=window_size)
        model = word2vec.Word2Vec.load(u'C:\\Users\\michar\\PycharmProjects\\play_gensim\\'+model_name)
        kmeans =  KMeans(n_clusters=NCLUSTERS, random_state=0).fit(model.syn0)

        score = hom_v_score(kmeans.labels_, ref_kmeans.labels_)[2]
        print "window {}, size {}, score {}".format(window_size, model_size, score)
        results[window_size].append((model_size, score))