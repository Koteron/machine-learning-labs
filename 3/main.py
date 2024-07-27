from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Загрузка изображения
image_path = 'Sea.png'
original_img = Image.open(image_path)

# Преобразование изображения в массив numpy
img_data = np.array(original_img)
pixels = img_data.reshape(-1, 3)  # Преобразование в список пикселей

# Количество кластеров цветов (сжатие до n цветов)
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)

# Получение центров кластеров (новые цвета)
new_colors = kmeans.cluster_centers_.astype(int)

# Присвоение каждому пикселю ближайшего цвета из кластера
new_pixels = new_colors[kmeans.labels_]

# Восстановление изображения
compressed_img_data = new_pixels.reshape(img_data.shape).astype("uint8")
compressed_img = Image.fromarray(compressed_img_data)

compressed_img.save("compressed_image.png","PNG")

# Отображение оригинального и сжатого изображений
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(original_img)
# ax[0].set_title('Original Image')
# ax[0].axis('off')
#
# ax[1].imshow(compressed_img)
# ax[1].set_title('Compressed Image')
# ax[1].axis('off')
#
# plt.show()