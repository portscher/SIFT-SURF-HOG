#!/usr/bin/env python3
from hog import HogTransformer
import os.path
import time
import sys
import argparse
import sys
import time

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import utils
from detect import Transformer
import preprocess


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))  # print classification report
    return accuracy_score(y_true, y_pred)  # return accuracy score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Method to use. Available: SIFT, SURF, HoG")
    parser.add_argument('-c', '--classes', nargs='+', help='<Required> Which classes to load', required=True)
    parser.add_argument('-k', '--k', type=int, default=100, help='Define number of clusters')
    parser.add_argument('-s', '--splits', type=int, default=3, help='Define number of KFold splits')
    # insert more meta-parameters here

    args = parser.parse_args()

    # TODO leave this in False. a bug still somewhere..
    UseCrossVal = True

    if not args.method:
        parser.print_help()
        sys.exit()

    # load images
    print("Loading classes " + ', '.join(args.classes) + "\n")
    images = utils.load_images('./img/', args.classes)
    X, Y = utils.separate_data(images)

    # extract features from training images
    print("Using " + args.method + " to extract features from the images..." + "\n")

    # TODO for now, only test_split

    print("using " + str(args.splits) + " splits")

    if UseCrossVal:
        cv = KFold(n_splits=args.splits, random_state=1, shuffle=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    feat = None
    if args.method.lower() == 'sift':
        feat = Transformer(args.k, args.method.lower())
    elif args.method.lower() == 'surf':
        feat = Transformer(args.k, args.method.lower())
    elif args.method.lower() == 'hog':
        feat = HogTransformer(args.k)
    else:
        raise Exception('No method', 'This method is not recognized')

    pipeline = Pipeline([
        ('preproc', preprocess.ImagePreparationTransformer()),
        ('feat', feat),
        ('svm', LinearSVC(max_iter=100000))  # ... set number of iterations higher than default (1000)
    ])

    if UseCrossVal:
        result = cross_val_score(pipeline, X, Y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score))
        print("Cross Validation score: \n")
        print(result)
        print("\nAvg accuracy: {}".format(result.mean()))
    else:
        print("Training with " + str(len(X_train)) + " samples")
        start = time.time()

        pipeline.fit(X_train, y_train)

        end = time.time()
        print("Training the model took " + str(end - start) + " seconds.")

        start = time.time()
        predictions = pipeline.predict(X_test)
        end = time.time()
        print("Predicting and feature extraction took " + str(end - start) + " seconds. \n\n")

        print(classification_report(y_test, predictions))

    sys.exit(0)


if __name__ == "__main__":
    main()
