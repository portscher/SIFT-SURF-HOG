import os.path

import cv2


def load_images(base, classes):
    """
    load all images of the given classes. each class corresponds to a folder in /img folder.
    :return: a list, containing key value pairs (class_id, image)
    """

    images = []
    for class_folder in classes:
        image_set = os.listdir(base + class_folder)
        labelled_images = [(cv2.imread(os.path.join(base, class_folder, image)), class_folder)
                           for image in image_set]
        images.extend(labelled_images)

    print("Loaded " + str(len(images)) + " images of the classes: " + ', '.join(classes))

    return images


def separate_data(images):
    """
    :param images: a list tuple (class_id, image)
    :return: X and Y
    """
    return zip(*images)


def join_data(data, labels):
    """
    :param data
    :param labels
    :return: list of tuples
    """

    return zip(labels, data)
