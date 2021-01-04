import cv2


def apply_surf(img, hessian=500, octave=4, octave_layers=2, ext=False):
    """
    Extract SURF keypoints and descriptors from a given image
    :param img: Image.
    :param hessian: Hessian Threshold, set to 500
    :param octave: Number of pyramid octaves the keypoint detector will use, set to 4
    :param octave_layers: Number of octave layers within each octave, set to 2
    :param ext: Extended descriptor flag, set to false
    :return: keypoints, descriptor
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


def get_descriptors_and_labels(func, img_set):
    """
    Applies a feature extraction algorithm on an image set.
    :param func: The respective feature extraction function
    :param img_set: The set on which the function is applied
    :return: Two lists of the same length: the descriptors and labels
    """
    descriptors = []
    labels = []

    for (img, lbl) in img_set:
        kp, des = func(img)
        for d in des:
            descriptors.append(d)
            labels.append(lbl)

    return descriptors, labels
