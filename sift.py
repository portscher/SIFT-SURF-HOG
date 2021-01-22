from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, BaseEstimator

import cv2
import numpy as np

class SiftTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, cluster_k):
        super(SiftTransformer, self).__init__()
        self.sift = cv2.SIFT_create()
        self.cluster_k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters=self.cluster_k, random_state=0, batch_size=6)
        self.features = []

    def fit(self, X, y=None, **kwargs):
        """
        Compute SIFT descriptors and learn clusters from data.
        :param X: features
        :param y: target vector
        :return: self: the class object
        """

        print("Identifying descriptors..")
        descriptors = []
        for img in X:
            key, desc = self.sift.detectAndCompute(img, None)
            descriptors.append(desc)

        descs = descriptors[0]

        for desc in descriptors[1:]:
            descs = np.vstack((descs, desc))

        print("Clustering data..")
        self.cluster.fit(descs)

        print("Clustered " + str(len(self.cluster.cluster_centers_)) + " centroids")

        return self

    def transform(self, X, y=None, **kwargs):
        """
        Compute histogram of X
        :param X: features
        :return: list of tupple histogram
        """

        print("Calculating histograms for " + str(len(X)) + " items.")
        histo_all = []
        for img in X:
            key, desc = self.sift.detectAndCompute(img, None)

            histo = np.zeros(self.cluster_k)
            nkp = np.size(len(key))

            for d in desc:
                idx = self.cluster.predict([d])
                # instead of increasing each bin by one, add the normalized value
                histo[idx] += 1/nkp

            histo_all.append(histo)

        return histo_all

    def fit_transform(self, X, y=None, **kwargs):
        print("fit and transforming..")

        self = self.fit(X, y)
        return self.transform(X, y)

    def get_histograms(self, images):
        """
        Calculate histograms for given images
        :param images: list of images
        :return: list of histograms
        """
        histo_all = []

        for img in images:
            key, desc = self.sift.detectAndCompute(img, None)
            histo = np.zeros(self.cluster_k)
            nkp = np.size(len(desc))

            for d in desc:
                idx = self.cluster.predict([d])
                # instead of increasing each bin by one, add the normalized value
                histo[idx] += 1/nkp

            histo_all.append(histo)

        return histo_all
