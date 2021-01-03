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
    img = cv2.resize(img, (350, 350))
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
    class_id = 0

    for class_folder in folders:
        image_set = os.listdir("img/" + class_folder)
        labelled_images = [(load_image("img/" + os.path.join(class_folder, image)), class_id) for image in image_set]
        random.shuffle(labelled_images)
        training_size = int(len(labelled_images) * 0.8)
        train_set.extend(labelled_images[- training_size:])
        test_set.extend(labelled_images[: - training_size or None])
        class_id += 1

    print("Loaded " + str(len(train_set)) + " images for training and " + str(
        len(test_set)) + " images for testing. Categories: " + ', '.join(folders))

    return train_set, test_set
