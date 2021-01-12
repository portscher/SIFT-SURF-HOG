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

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (340, 350))
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
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
        labelled_images = [(load_image("img/" + os.path.join(class_folder, image)), class_folder) for image in image_set]
        random.shuffle(labelled_images)
        training_size = int(len(labelled_images) * 0.8)
        train_set.extend(labelled_images[- training_size:])
        test_set.extend(labelled_images[: - training_size or None])

    print("Loaded " + str(len(train_set)) + " images for training and " + str(
        len(test_set)) + " images for testing. Categories: " + ', '.join(folders))

    return train_set, test_set

def load_images(classes):
    """
    load all images of the given classes. each class correspond to a folder in the img folder.
    :return: a list, containing key value pairs (class_id, image)
    """

    images = []
    for class_folder in classes:
        image_set = os.listdir("img/" + class_folder)
        labelled_images = [(class_folder, load_image("img/" + os.path.join(class_folder, image)))
                for image in image_set]
        images.extend(labelled_images)

    print("Loaded " + str(len(images)) + " images of the classes: " + ', '.join(classes))

    return images

def encode_classes(classid):
    """
    encode the class-id into an integer
    :param classid: name of the class
    :return: an integer identifier for that class
    """

    val = 0
    if classid == 'cactus':
        val = 0
    if classid == 'dice':
        val = 1
    if classid == 'raccoon':
        val = 2
    if classid == 'spaghetti':
        val = 3
    if classid == 'sushi':
        val = 4

    return val

def separate_data(images):
    """
    :param images: a list tuple (classid, image)
    :return: X and Y
    """
    Y, X = zip(*images)

    Y = map(encode_classes, Y)

    return X, list(Y)

def join_data(X, Y):
    """
    :param X: data
    :param Y: labels
    :return: list of tuple (Y, X)
    """

    return zip(Y, X)
