import os.path

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import feature_extraction
import utils

import time


def main():
    # load images
    train_set, test_set = utils.load_data(os.listdir("img"))

    # extract features from training images
    print("Using SURF to extract features from the images...")
    train_descriptors = []
    train_labels = []
    for (img, lbl) in train_set:
        kp, des = feature_extraction.apply_surf(img)
        for d in des:
            train_descriptors.append(d)
            train_labels.append(lbl)

    print("Training the model...")
    start = time.time()
    svm = LinearSVC()
    svm.fit(train_descriptors, train_labels)
    end = time.time()

    print("Training the model took " + str(end - start) + " seconds.")

    # extract features from test images
    test_descriptors = []
    test_labels = []

    for (img, lbl) in test_set:
        kp, des = feature_extraction.apply_surf(img)
        for d in des:
            test_descriptors.append(d)
            test_labels.append(lbl)

    # test the model and print report
    predictions = svm.predict(test_descriptors)
    print(classification_report(test_labels, predictions))


if __name__ == "__main__":
    main()
