import math
import random

import numpy as np
from tensorflow.keras.utils import Sequence
from myutils.image_processor import preprocess_image


class MyTrainDataSequence(Sequence):
    def __init__(
            self,
            paths_list=None,
            labels_list=None,
            batch_size=None,
            target_height=None,
            target_width=None,
            rescale=None,
    ):
        self.paths_list = paths_list
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.rescale = rescale
        self.preprocess = preprocess_image

    def __len__(self):
        return math.ceil((len(self.paths_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths_list = self.paths_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels_list = self.labels_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_images_list = [
            self.preprocess(
                bpl,
                target_height=self.target_height,
                target_width=self.target_width,
                rescale=self.rescale,
            ) for bpl in batch_paths_list
        ]

        return np.array(batch_images_list), np.array(batch_labels_list)

    def on_epoch_end(self):
        if self.shuffle:
            zipped_list = [item for item in zip(self.paths_list, self.labels_list)]
            random.shuffle(zipped_list)
            self.paths_list, self.labels_list = zip(*zipped_list)


class MyTestDataSequence(Sequence):
    def __init__(
            self,
            paths_list=None,
            batch_size=None,
            target_height=None,
            target_width=None,
            rescale=None,
            batch_mode=False
    ):
        self.paths_list = paths_list
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.rescale = rescale
        self.batch_mode = batch_mode
        self.preprocess = preprocess_image

    def __len__(self):
        return math.ceil((len(self.paths_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths_list = self.paths_list[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_images_list = [
            self.preprocess(
                bpl,
                target_height=self.target_height,
                target_width=self.target_width,
                rescale=self.rescale,
                batch_mode=self.batch_mode
            ) for bpl in batch_paths_list
        ]
        print("\r", idx, " batches imgs have been processed. ", end="", flush=True)
        return np.array(batch_images_list)

    def set_paths_list(self, paths_list):
        self.paths_list = paths_list
