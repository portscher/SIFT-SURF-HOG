import utils
import feature_extraction
import os.path
import cv2


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

    kp, des = feature_extraction.apply_surf(sushi_training_images[0])
    images, all_descriptors = feature_extraction.get_class_descriptors(sushi_training_images,
                                                                       feature_extraction.apply_surf)

    result = sushi_training_images[0].copy()
    result = cv2.drawKeypoints(result, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
