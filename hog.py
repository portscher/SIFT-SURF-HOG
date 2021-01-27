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
        self.orientations = 9
        self.pixelsPerCells = (8, 8)
        self.cellsPerBlock = (2, 2)
        self.BlockNorm = 'L2-Hys'
        self.TransformSqrt = True
        self.FeatureVector = True

        self.cluster_k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters = cluster_k, init_size=3 * cluster_k, random_state = 0, batch_size = 6)
        self.features = []
    
    def fit(self, X, y = None, **kwargs):
        """
        Learns the cluster by computing HoG feature descriptors.
        """
        print("Identify descriptors...");

        descriptors = []
        for img in self.prepareData(X):
            fd = hog(img, self.orientations, self.pixelsPerCells, self.cellsPerBlock, self.BlockNorm, transform_sqrt=self.TransformSqrt, feature_vector=self.FeatureVector)
            descriptors.append(fd)
        
        print("Clustering data...")
        self.cluster.fit(descriptors)

        print("Clustered " + str(len(self.cluster.cluster_centers_)) + " centroids")

        return self
    
    def transform(self, X, y=None, **kwargs):
        """
        Creates the histograms for the given data.
        The HoG implementation of skimage returns the
        feature descriptor of HoG. Therefore the feature descriptor
        is equal to the produced histogram of oriented gradients.
        """
        print("Calculating histograms for " + str(len(X)) + " items.")
        
        histograms = []
        for img in self.prepareData(X):
            fd = hog(img, self.orientations, self.pixelsPerCells, self.cellsPerBlock, self.BlockNorm, transform_sqrt=self.TransformSqrt, feature_vector=self.FeatureVector)
            histograms.append(fd)
        
        return histograms
    
    def prepareData(self, X):
        """
        Prepares the data for the HoG transformer.
        The images will be resized to a aspect ratio of 1 : 2 (height : width)
        and will also be converted to gray images (no RGB colors).
        These steps results in a better performance of the HoG feature, according to
        the source https://learnopencv.com/histogram-of-oriented-gradients/
        """
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