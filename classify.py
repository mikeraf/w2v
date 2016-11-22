from gensim.models import word2vec
from sklearn.cluster import KMeans

#cd PycharmProjects/play_gensim/

w2v = word2vec.Word2Vec.load(u'C:\\Users\\michar\\PycharmProjects\\play_gensim\\w2v_model_size_100_window_5.model')

kmeans = KMeans(n_clusters=50, random_state=0).fit(w2v.syn0)


labels_count = {}
for l in kmeans.labels_:
    labels_count[l] = labels_count.get(l,0)+1
for l in kmeans.labels_:
    labels_count[l] = labels_count.get(l,0)+1

sorted_labels = sorted(labels_count.keys(), key = labels_count.get)

minimal_label = sorted_labels[2]
for i,l in enumerate(kmeans.labels_):
    if l == minimal_label:
        print w2v.index2word[i]
