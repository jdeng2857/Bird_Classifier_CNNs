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

    def __init__(self, use_rows, batch_size, csv_file="birds.csv"):
        self.csv_df = pd.read_csv(csv_file)
        self.use_rows = use_rows
        self.batch_size = batch_size
        self.dataset = pd.DataFrame(self.csv_df).to_numpy()

        self.labels = self.dataset[:,2]
        self.img_filepaths = self.dataset[:,1]
        self.class_ids = self.dataset[:,0]

        (x,) = self.img_filepaths.shape
        print(self.dataset.shape)
        print(self.img_filepaths[0])
        image = Image.open(self.img_filepaths[0])
        image_array = np.expand_dims(np.asarray(image), axis=0)
        print(image_array.shape)

    def __len__(self):
        # batches_per_epoch is the total number of batches used for one epoch
        batches_per_epoch = int(len(self.use_rows) / self.batch_size)
        return batches_per_epoch

    def __getitem__(self, index):
        # index is the index of the batch to be retrieved
        batch_ids = self.use_rows[index * self.batch_size: (index + 1) * self.batch_size]

        x = None
        y = None

        for curr_id in batch_ids:
            image = Image.open(self.img_filepaths[curr_id]).resize((224,224))
            image_array = np.expand_dims(np.asarray(image),axis=0)

            if x is None:
                x = image_array
                y = self.class_ids[curr_id]
            else:
                x = np.vstack((x, image_array))
                y = np.vstack((y, self.class_ids[curr_id]))

        y = tf.keras.utils.to_categorical(y, num_classes=511)
        return x, y

use_rows = list(range(0,87050))
bird_data_generator = BirdDataGenerator(csv_file="birds.csv", use_rows=use_rows, batch_size=128)
bird_data_generator.__getitem__(0)

def bird_cnn_model(csv_file):
    model = Sequential([
        InputLayer(input_shape=(224,224,3)),
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(511, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer = optimizer, metrics=["accuracy"])

    print(model.summary())
    train_rows = list(range(0,81950))
    train_generator = BirdDataGenerator(csv_file=csv_file, use_rows=train_rows, batch_size=64)
    h = model.fit(x=train_generator, epochs=10, verbose=1)
    print(h)

    test_rows = list(range(81950, 84500))
    test_generator = BirdDataGenerator(csv_file=csv_file, use_rows=test_rows, batch_size=128)
    model.evaluate(x=test_generator)

    return model

one_epoch_model = tf.keras.models.load_model("batch_normalized_1_epoch")
one_epoch_model.summary()
test_image = Image.open("images to predict/1.jpg").resize((224,224))
image_array = np.expand_dims(np.asarray(test_image),axis=0)
print(image_array)
predict_x = one_epoch_model.predict(image_array)
classes_x = np.argmax(predict_x, axis=1)
print(classes_x)

# Generate class labels
csv_df = pd.read_csv("birds.csv")
dataset = pd.DataFrame(csv_df).to_numpy()
labels = dataset[:,2]
class_ids = dataset[:,0]

print(labels.shape)
print(class_ids.shape)

label_mapping = {}
id_to_name = dict(zip(class_ids, labels))
print(id_to_name)
print(id_to_name[classes_x[0]])
