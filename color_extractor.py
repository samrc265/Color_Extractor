import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_colors(image_path, num_colors=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))  # Flatten to 2D array

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(image)

    colors = np.round(kmeans.cluster_centers_).astype(int)
    return [tuple(color) for color in colors]
