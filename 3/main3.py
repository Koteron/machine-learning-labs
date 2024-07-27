import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('votes.csv')

# Обработка пропущенных значений
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Масштабирование данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Применение Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=5)
agg_clustering.fit(data_scaled)

# Получение linkage matrix для построения дендрограммы
linkage_matrix = linkage(data_scaled, method='ward')

# Построение дендрограммы
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix,
           labels=agg_clustering.labels_,
           leaf_rotation=90,
           leaf_font_size=12)
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()