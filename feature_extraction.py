import cv2
import numpy as np


def apply_surf(img, hessian=500, octave=4, octave_layers=2, ext=False):
    """
    Extract SURF keypoints and descriptors from a given image
    :param img: Image.
    :param hessian: Hessian Threshold, set to 500
    :param octave: Number of pyramid octaves the keypoint detector will use, set to 4
    :param octave_layers: Number of octave layers within each octave, set to 2
    :param ext: Extended descriptor flag, set to false
    :return:
    """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian, nOctaves=octave, nOctaveLayers=octave_layers,
                                       extended=ext)
    return surf.detectAndCompute(img, None)


def apply_sift(img, octave=3, contrast=0.03, edge=10, sigma=1.6):
    """
    Extract SIFT keypoints and descriptors from a given image
    """
    #  TODO


def apply_hog(img):
    """
    Apply HoG to the image
    """
    #  TODO


def get_class_descriptors(image_class, feature_detection_function):
    """
    Get all descriptors for an image class.
    Images with too few keypoints are filtered out here.
    :param image_class: Set of images
    :param feature_detection_function: the respective feature descriptor function
    :return: All images with enough keypoints, and descriptors
    """
    all_descriptors = None
    images = []
    for img in image_class:
        kp, des = feature_detection_function(img)
        if len(kp) < 500:  # TODO: Which value makes sense here?
            continue

        if all_descriptors is None:
            all_descriptors = des
        else:
            all_descriptors = np.concatenate([all_descriptors, des], 0)
        images.append(img)

    print("Keeping " + str(len(images)) + " images of " + str(len(image_class)))
    return images, all_descriptors
