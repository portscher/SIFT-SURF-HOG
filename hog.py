from skimage import feature
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, BaseEstimator

import cv2
import numpy as np

class HogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, cluster_k):
        super(HogTransformer, self).__init__()
        self.cluster_k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters = cluster_k, init_size=3 * cluster_k, random_state = 0, batch_size = 6)
        self.features = []
    
    def fit(self, X, y = None, **kwargs):

        print("Identify descriptors...");
        descriptors = []
        for img in self.prepareData(X):
            fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
            descriptors.append(fd)
        
        print("Clustering data...")
        self.cluster.fit(descriptors)

        print("Clustered " + str(len(self.cluster.cluster_centers_)) + " centroids")

        return self
    
    def transform(self, X, y=None, **kwargs):
        print("Calculating histograms for " + str(len(X)) + " items.")
        
        histograms = []
        for img in self.prepareData(X):
            fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, feature_vector=True)
            histograms.append(fd)
        
        return histograms
    
    def prepareData(self, X):
        print("Prepare data for HoG")

        preparedData = []
        for img in X:
            (height, _, _) = img.shape
            resizedImage = cv2.resize(img, (height, height * 2))
            preparedData.append(cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY))

        return preparedData
    
    def fit_transform(self, X, y=None, **kwargs):
        print("fit and transforming...")

        
        self = self.fit(X, y)
        return self.transform(X, y)