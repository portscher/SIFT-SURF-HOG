import os.path
import random

import cv2


def load_image(image_path):
    """
    loads an image
    :param image_path: location of the image
    :return: resized and normalized image in grayscale
    """
    while not os.path.isfile(image_path):
        pass

    img = cv2.imread(image_path)
    img = cv2.resize(img, (340, 350))
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def load_data(folders):
    """
    loads a whole set of images, in random order, and labels them according to their image class
    :param folders: Name of the folder that contains the images
    :return: 2 lists, each containing key value pairs (image, class_id)
            one list of all training images (80% of all)
            one list of all testing images (20% of all)
    """
    train_set = []
    test_set = []

    for class_folder in folders:
        image_set = os.listdir("img/" + class_folder)
        labelled_images = [(load_image("img/" + os.path.join(class_folder, image)), class_folder) for image in
                           image_set]
        random.shuffle(labelled_images)
        training_size = int(len(labelled_images) * 0.8)
        train_set.extend(labelled_images[- training_size:])
        test_set.extend(labelled_images[: - training_size or None])

    print("Loaded " + str(len(train_set)) + " images for training and " + str(
        len(test_set)) + " images for testing. Categories: " + ', '.join(folders))

    return train_set, test_set

def load_images(base, classes):
    """
    load all images of the given classes. each class corresponds to a folder in /img folder.
    :return: a list, containing key value pairs (class_id, image)
    """

    images = []
    for class_folder in classes:
        image_set = os.listdir(base + "img/" + class_folder)
        labelled_images = [(load_image("img/" + os.path.join(class_folder, image)), class_folder)
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
