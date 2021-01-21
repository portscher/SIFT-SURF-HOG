#!/usr/bin/env python3
import argparse
import sys
import time

import cv2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import utils
from sift import Transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method to use. Available: SIFT, SURF, HoG")
    parser.add_argument('-c', '--classes', nargs='+', help='<Required> Which classes to load', required=True)
    parser.add_argument('-k', '--k', type=int, default=100, help='Define number of clusters')
    # insert more meta-parameters here

    args = parser.parse_args()
    print(args.method)

    if not args.method:
        parser.print_help()
        sys.exit()

    # load images
    print("Loading classes " + ', '.join(args.classes))
    images = utils.load_images(args.classes)
    X, Y = utils.separate_data(images)

    # extract features from training images
    print("Using " + args.method + " to extract features from the images...")

    # TODO for now, only test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    feat = None
    if args.method.lower() == 'sift':
        feat = Transformer(args.k, cv2.SIFT_create)
    elif args.method.lower() == 'surf':
        feat = Transformer(args.k, cv2.xfeatures2d.SURF_create)
    elif args.method.lower() == 'hog':
        raise Exception('TODO', 'Not yet implemented')
        feat = Transformer(args.k)
    else:
        raise Exception('No method', 'This method is not recognized')

    pipeline = Pipeline([
        ('feat', feat),
        ('svm', LinearSVC(max_iter=50000))  # ... set number of iterations higher than default (1000)
    ])

    print("Training with " + str(len(X_train)) + " samples")
    start = time.time()

    pipeline.fit(X_train, y_train)

    end = time.time()
    print("Training the model took " + str(end - start) + " seconds.")

    # TODO use cross_validate
    print("\n====> average score: " + str(pipeline.score(X_test, y_test)) + "\n")

    start = time.time()
    predictions = pipeline.predict(X_test)
    end = time.time()
    print("Predicting and feature extraction took " + str(end - start) + " seconds. \n\n")

    print(classification_report(y_test, predictions))

    sys.exit(0)


if __name__ == "__main__":
    main()
