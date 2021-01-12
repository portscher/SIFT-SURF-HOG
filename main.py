#!/usr/bin/env python3
import os.path
import time
import sys
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import feature_extraction
import utils

from sift import SiftTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method to use. Available: SIFT, SURF, HoG")
    parser.add_argument('-c','--classes', nargs='+', help='<Required> Which classes to load', required=True)
    parser.add_argument('-k','--k', type=int, default=100, help='Define number of clusters')
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
    print("Using "+ args.method +" to extract features from the images...")

    # TODO for now, only test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

    feat = None
    if args.method.lower() == 'sift':
        feat = SiftTransformer(args.k)
    elif args.method.lower() == 'suft':
        raise Exception('TODO', 'Not yet implemented')
        feat = SiftTransformer(args.k)
    elif args.method.lower() == 'hog':
        raise Exception('TODO', 'Not yet implemented')
        feat = SiftTransformer(args.k)
    else:
        raise Exception('No method', 'This method is not recognized')


    pipeline = Pipeline([
        ('feat', feat),
        ('svm', LinearSVC())
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
