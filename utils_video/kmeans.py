# divide the image into k clusters
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils import shuffle


# convert image to numpy array
def img2array(img):
    return np.array(img)


# k is unknown
def kmeans(X, k_init=10, max_iter=100):
    # k_init = 10
    # max_iter = 100
    # shuffle the data
    X = shuffle(X, random_state=0)
    # kmeans = KMeans(n_clusters=k_init, random_state=0).fit(X)
    kmeans = KMeans(n_clusters=k_init, random_state=0, max_iter=max_iter).fit(X)
    return kmeans


# main function
if 
