import cv2
import os.path
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


def main():
    # load cactus images
    cactus_folder = 'img/cactus'
    cactus_training_images, cactus_test_images = utils.load_set_of_images(cactus_folder, os.listdir(cactus_folder))

    # load dice images
    dice_folder = 'img/dice'
    dice_training_images, dice_test_images = utils.load_set_of_images(dice_folder, os.listdir(dice_folder))

    # load raccoon images
    raccoon_folder = 'img/raccoon'
    raccoon_training_images, raccoon_test_images = utils.load_set_of_images(raccoon_folder, os.listdir(raccoon_folder))

    # load spaghetti images
    spaghetti_folder = 'img/spaghetti'
    spaghetti_training_images, spaghetti_test_images = utils.load_set_of_images(spaghetti_folder,
                                                                                os.listdir(spaghetti_folder))

    # load sushi images
    sushi_folder = 'img/sushi'
    sushi_training_images, sushi_test_images = utils.load_set_of_images(sushi_folder, os.listdir(sushi_folder))

    kp, des = apply_surf(sushi_training_images[0])
    images, all_descriptors = get_class_descriptors(sushi_training_images)

    result = sushi_training_images[0].copy()
    result = cv2.drawKeypoints(result, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


if __name__ == "__main__":
    main()
