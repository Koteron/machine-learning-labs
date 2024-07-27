import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score

# Загрузка данных
data = pd.read_csv('clustering_2.csv', header=None, delimiter='\t')
data.columns = ['X', 'Y']

# DBI_kmeans = []
# for k in range(2, 14):
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
#     DBI_kmeans.append(davies_bouldin_score(data, kmeans.labels_))
#
# plt.plot(range(2,14), DBI_kmeans, marker='o')
# plt.title("K-means")
# plt.xlabel("Cluster number")
# plt.ylabel("DBI")
# plt.show()
#
#
#
# DBI_agglomerative = []
# for k in range(2, 14):
#     agg_clustering = AgglomerativeClustering(n_clusters=k)
#     labels = agg_clustering.fit_predict(data)
#     DBI_agglomerative.append(davies_bouldin_score(data, agg_clustering.labels_))
#
# plt.plot(range(2,14), DBI_agglomerative, marker='o')
# plt.title("Agglomerative Clustering")
# plt.xlabel("Cluster number")
# plt.ylabel("DBI")
# plt.show()


data_x = list()
data_y = list()
#data_x, data_y = data.to_numpy()
for row in data.to_numpy():
    data_x.append(row[0])
    data_y.append(row[1])
plt.scatter(data_x, data_y)
plt.show()

#kmeans = KMeans(n_clusters=6, random_state=0).fit(data)
# dbscan = DBSCAN()
# dbscan.fit(data)
# data['KMeans_labels'] = dbscan.labels_
# plt.scatter(data['X'], data['Y'], c=data['KMeans_labels'], cmap='viridis')
# plt.title('clustering_3.csv')
# plt.show()