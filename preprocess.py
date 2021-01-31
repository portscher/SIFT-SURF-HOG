import cv2
from sklearn.base import BaseEstimator, TransformerMixin


class ImagePreparationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super(ImagePreparationTransformer, self).__init__()

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None, **kwargs):
        images = []
        for img in X:
            img = cv2.resize(img, (340, 350))
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            images.append(img)
        return images

    def fit_transform(self, X, y=None, **kwargs):
        print("Pre-processing images..")
        self = self.fit(X, y)
        return self.transform(X, y)
