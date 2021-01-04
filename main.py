import os.path
import time

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import feature_extraction
import utils


def main():
    # load images
    train_set, test_set = utils.load_data(os.listdir("img"))

    # extract features from training images
    print("Using SURF to extract features from the images...")

    train_descriptors, train_labels = feature_extraction.get_descriptors_and_labels(feature_extraction.apply_surf, train_set)

    print("Training the model...")
    start = time.time()

    svm = LinearSVC()
    svm.fit(train_descriptors, train_labels)

    end = time.time()

    print("Training the model took " + str(end - start) + " seconds.")

    # extract features from test images
    test_descriptors, test_labels = feature_extraction.get_descriptors_and_labels(feature_extraction.apply_surf, test_set)

    # test the model and print report
    predictions = svm.predict(test_descriptors)
    print(classification_report(test_labels, predictions))


if __name__ == "__main__":
    main()
