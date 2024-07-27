import pandas as pd
import sklearn.metrics

# Load your data
data = pd.read_csv('pluton.csv')
print(data.head())
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Function to perform k-means and plot results
# def perform_kmeans(data, max_iter, title):
#     kmeans = KMeans(n_clusters=3, max_iter=max_iter, random_state=42)
#     clusters = kmeans.fit_predict(data)
#     print(f"Inertia for {title}: {kmeans.inertia_}")
#
#     # Plotting the clusters
#     # plt.figure(figsize=(8, 4))
#     # plt.scatter(data[:, 0], data[:, 2], c=clusters, cmap='viridis', marker='o')
#     # plt.title(title)
#     # plt.xlabel('Pu238')
#     # plt.ylabel('Pu240')
#     # plt.colorbar()
#     # plt.show()

dbi_original = list()
iter_arr = list()
for iter in range (2, 15):
    iter_arr.append(iter)
    kmeans = KMeans(n_clusters=3, max_iter=500, random_state=42)
    clusters = kmeans.fit_predict(data.values)
    dbi_original.append(davies_bouldin_score(data.values, kmeans.labels_))

dbi_scaled = list()
for iter in range (2, 15):
    kmeans = KMeans(n_clusters=3, max_iter=500, random_state=42)
    clusters = kmeans.fit_predict(data_scaled)
    dbi_scaled.append(davies_bouldin_score(data_scaled, kmeans.labels_))

plt.plot(iter_arr, dbi_original)
plt.show()
plt.plot(iter_arr, dbi_scaled)
plt.show()