import cv2

import numpy as np
import utils

hessian_threshold = 500
surf = cv2.xfeatures2d.SURF_create(hessian_threshold)


def apply_surf(img):
    return surf.detectAndCompute(img, None)


def get_class_descriptors(image_class):
    all_descriptors = None
    images = []
    for img in image_class:
        kp, des = apply_surf(img)
        # ignore images with too few key points
        if len(kp) < hessian_threshold:
            continue

        if all_descriptors is None:
            all_descriptors = des
        else:
            all_descriptors = np.concatenate([all_descriptors, des], 0)
        images.append(img)

    print("Keeping " + str(len(images)) + " images of " + str(len(image_class)))
    return images, all_descriptors
