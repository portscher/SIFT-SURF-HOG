from numpy.core.shape_base import block
from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, BaseEstimator

import cv2
import numpy as np

class HogTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, cluster_k):
        winSize = (64,128)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        super(HogTransformer, self).__init__()
        self.hogDescriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        self.k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters = cluster_k, random_state = 0, batch_size = 6)
        self.features = []
    
    def fit(self, X, y = None, **kwargs):

        print("Identify descriptors...");
        descriptors = []
        images = []
        for img in X:
            resizedImage = cv2.resize(img, (64, 128))
            grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
            images.append(grayImage)

        for img in images:
            fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
            descriptors.append(fd)

        descriptorList = descriptors[0]
        for desc in descriptors[1:]:
            descriptorList = np.vstack((descriptorList, desc))
        
        print("Clustering data...")
        self.cluster.fit(descriptorList)

        print("Clustered " + str(len(self.cluster.cluster_centers_)) + " centroids")

        return self
    
    def transform(self, X, y=None, **kwargs):
        print("Calculating histograms for " + str(len(X)) + " items.")
        allHistograms = []
        images = []
        for img in X:
            resizedImage = cv2.resize(img, (64, 128))
            grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
            images.append(grayImage)

        for img in images:
            cellSize = (8, 8)
            blockSize = (2, 2)
            nbins = 9
            nCells = (img.shape[0] // cellSize[0], img.shape[1] // cellSize[1])
            allHistograms.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True, transform_sqrt=True, multichannel=True))
        
        return allHistograms
    
    def fit_transform(self, X, y=None, **kwargs):
        print("fit and transforming...")

        self = self.fit(X, y)
        return self.transform(X, y)