import cv2
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import MiniBatchKMeans


class Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, cluster_k, detector_str):
        super(Transformer, self).__init__()

        # we need this dummy variable!
        self.detector_str = detector_str

        if detector_str.lower() == 'sift':
            self.detector = cv2.SIFT_create(
                nfeatures=0,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        elif detector_str.lower() == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=100,
                nOctaves=4,
                nOctaveLayers=3,
                extended=False,
                upright=False)
        self.cluster_k = cluster_k
        self.cluster = MiniBatchKMeans(n_clusters=cluster_k, init_size=3 * cluster_k, random_state=0, batch_size=6)
        self.features = []

    # def determine_optimal_k(self, X):
    #
    #     descriptors = []
    #     for img in X:
    #         key, desc = self.detector.detectAndCompute(img, None)
    #         descriptors.append(desc)
    #
    #     descs = descriptors[0]
    #
    #     for desc in descriptors[1:]:
    #         descs = np.vstack((descs, desc))
    #
    #     distortions = []
    #     K = range(100, 600, 50)
    #     for k in K:
    #         cluster = MiniBatchKMeans(n_clusters=k)
    #         cluster.fit(descs)
    #         distortions.append(cluster.inertia_)
    #
    #     plt.figure(figsize=(16, 8))
    #     plt.plot(K, distortions, 'bx-')
    #     plt.xlabel('k')
    #     plt.ylabel('Distortion')
    #     plt.title('The Elbow Method showing the optimal k')
    #     plt.show()

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
            key, desc = self.detector.detectAndCompute(img, None)
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
        :return: list of tuple histogram
        """

        print("Calculating histograms for " + str(len(X)) + " items.")
        histo_all = []
        for img in X:
            key, desc = self.detector.detectAndCompute(img, None)

            histo = np.zeros(self.cluster_k)
            nkp = np.size(len(key))

            for d in desc:
                idx = self.cluster.predict([d])
                # instead of increasing each bin by one, add the normalized value
                histo[idx] += 1 / nkp

            histo_all.append(histo)

        return histo_all

    def fit_transform(self, X, y=None, **kwargs):
        print("fitting and transforming..")

        # self.determine_optimal_k(X)

        self = self.fit(X, y)
        return self.transform(X, y)
