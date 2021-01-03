import cv2
import os.path
import random


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


def load_set_of_images(folder_name, file_names):
    """
    loads a whole set of images, in random order
    :param folder_name: Name of the folder that contains the images
    :param file_names: List of the files to be loaded
    :return: one list of all training images (80% of all), one list of all testing images (20% of all)
    """
    images = [load_image(os.path.join(folder_name, file_name)) for file_name in file_names]
    random.shuffle(images)

    training_size = int(len(images) * 0.8)

    train_set = images[- training_size:]
    test_set = images[: - training_size or None]

    print("Loaded " + str(len(images)) + " images from " + folder_name + ", " + str(
        training_size) + " for training and " + str(len(images) - training_size) + " for testing.")

    return train_set, test_set
