
import cv2 as cv
import tensorflow as tf


def preprocess_image(img_path,
                     target_height=None,
                     target_width=None,
                     rescale=None,
                     batch_mode=None):
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

    if img is None:
        print("No image file found!")
        return -1
    else:
        height = img.shape[0]
        width = img.shape[1]
        if target_height and target_width:
            if target_height != height or target_width != width:
                img = cv.resize(img, (target_width, target_height))

        mean = cv.mean(img)
        img = cv.subtract(img, mean)

        if rescale:
            img = img * rescale

        if img.ndim == 2:
            img = img[..., tf.newaxis]

        if batch_mode:
            img = img[tf.newaxis, ...]

        return img
