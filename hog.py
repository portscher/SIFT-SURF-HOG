from skimage.feature import hog
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin, BaseEstimator

import cv2


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Provides functionality to use the HoG for training clusters
    and computing histograms of oriented gradients.
    """

    def __init__(self, cluster_k):
        super(HogTransformer, self).__init__()
        # Default values for the HoG according to the values given at the paper of Navneet Dalal and Bill Triggs.
        # https://hal.inria.fr/inria-00548512/document
        self.cellsPerBlock = (2, 2)
        self.TransformSqrt = True

        self.cluster_k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters=cluster_k, init_size=3 * cluster_k, random_state=0, batch_size=6)
        self.features = []

    def fit(self, X, y=None, **kwargs):
        """
        Learns the cluster by computing HoG feature descriptors.
        """
        print("Identify descriptors...");

        descriptors = []
        for img in self.prepareData(X):
            # we must only change the cells per block to (2, 2) and the transform sqrt to true to be conform with the paper.
            fd = hog(img, cells_per_block=self.cellsPerBlock, block_norm='L2-Hys', transform_sqrt=self.TransformSqrt)
            descriptors.append(fd)

        print("Clustering data...")
        self.cluster.fit(descriptors)

        print("Clustered " + str(len(self.cluster.cluster_centers_)) + " centroids")

        return self

    def transform(self, X, y=None, **kwargs):
        """
        Creates the histograms for the given data.
        The HOG implementation of skimage returns the
        feature descriptor of HOG. Therefore the feature descriptor
        is equal to the produced histogram of oriented gradients.
        """
        print("Calculating histograms for " + str(len(X)) + " items.")

        histograms = []
        for img in self.prepareData(X):
            fd = hog(img, cells_per_block=self.cellsPerBlock, block_norm='L2-Hys', transform_sqrt=self.TransformSqrt)
            histograms.append(fd)

        return histograms

    def prepareData(self, X):
        """
        Prepares the data for the HOG transformer.
        The images will be resized to 64 by 128 pixels, according to the paper of Navneet Dalal and Bill Triggs.
        This step results in a better performance of the HOG feature, according to
        the source https://learnopencv.com/histogram-of-oriented-gradients/
        """
        print("Prepare data for HoG")

        preparedData = []
        for img in X:
            preparedData.append(cv2.resize(img, (64, 128)))

        return preparedData

    def fit_transform(self, X, y=None, **kwargs):
        print("fit and transforming...")

        self = self.fit(X, y)
        return self.transform(X, y)
