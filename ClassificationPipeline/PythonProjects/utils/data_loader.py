import random

import numpy as np
from tensorflow.keras.utils import to_categorical
from utils.DataSequence import MyTrainDataSequence
from myutils.file_processor import get_files


def load_data_list(
        data_dir=None,
        shuffle=True
):
    data_list = get_files(data_dir, suffix='.jpg')

    data_paths_list = []
    data_labels_list = []
    label_flag = 0.
    for item in data_list:
        current_dir = item[0]
        image_name_list = item[1]
        image_paths_list = [current_dir + '\\' + image_name for image_name in image_name_list]
        labels_list = np.zeros(len(image_paths_list)) + label_flag
        label_flag += 1.

        data_paths_list.extend(image_paths_list)
        data_labels_list.extend(labels_list)

    if shuffle:
        zipped_list = [item for item in zip(data_paths_list, data_labels_list)]
        random.shuffle(zipped_list)
        data_paths_list, data_labels_list = zip(*zipped_list)

    data_labels_list = to_categorical(data_labels_list)
    return np.array(data_paths_list), np.array(data_labels_list)


def load_data_as_sequence(
        paths_list,
        labels_list,
        batch_size=32,
        target_height=0,
        target_width=0,
        rescale=1 / 255.
):
    return MyTrainDataSequence(
        paths_list=paths_list,
        labels_list=labels_list,
        batch_size=batch_size,
        target_height=target_height,
        target_width=target_width,
        rescale=rescale,
    )