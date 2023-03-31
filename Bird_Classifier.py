import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, BatchNormalization, Conv2D, MaxPooling2D, Dropout
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from PIL import Image


class BirdDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, csv_file, use_rows, batch_size):
        self.csv_df = pd.read_csv(csv_file)
        self.use_rows = use_rows
        self.batch_size = batch_size
        self.dataset = pd.DataFrame(self.csv_df).to_numpy()

        self.labels = self.dataset[:, 2]
        self.img_filepaths = self.dataset[:, 1]
        self.class_ids = self.dataset[:, 0]

        (x,) = self.img_filepaths.shape

        print(self.img_filepaths[0])
        image = Image.open(self.img_filepaths[0])
        image_array = np.expand_dims(np.asarray(image), axis=0)

        print(image_array)
        print(image_array.shape)

    #         self.images = []

    #         for i in range(0, 10):
    #             image = Image.open(self.img_filepaths[0])
    #             image_array = np.expand_dims(np.asarray(image), axis=0)
    #             self.images.append(image_array)

    #         self.images = np.array(self.images)

    #         print(self.images.shape)

    def __len__(self):
        # batches_per_epoch is the total number of batches used for one epoch
        batches_per_epoch = int(len(self.use_rows) / self.batch_size)
        return batches_per_epoch

    def __getitem__(self, index):
        # index is the index of the batch to be retrieved
        batch_ids = self.use_rows[index * self.batch_size: (index + 1) * self.batch_size]

        x = None
        y = None

        images = []
        labels = []

        for curr_id in batch_ids:
            image = Image.open(self.img_filepaths[curr_id])
            img_array = np.expand_dims(np.asarray(image), axis=0)
            images.append(img_array)
            labels.append(self.labels[curr_id])

        images = np.array(images)
        labels = np.array(labels)
        print(images.shape)
        print(labels.shape)

        x = images
        y = labels

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.split_ids)

use_rows = list(range(0,87050))
bird_data_generator = BirdDataGenerator(csv_file="birds.csv", use_rows=use_rows, batch_size=128)
bird_data_generator.__getitem__(0)