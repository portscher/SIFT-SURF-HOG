import os.path

import cv2

import feature_extraction
import utils


def main():
    # load images
    train_set, test_set = utils.load_data(os.listdir("img"))

    for (img, lbl) in train_set:
        kp, des = feature_extraction.apply_surf(img)
        # TODO

    # Train the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))


if __name__ == "__main__":
    main()
