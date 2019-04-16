import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, \
    MaxPool2D, ReLU
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


class SiameseNet(Model):
    """ Implementation for Siamese networks. """

    def __init__(self, w, h, c):
        super(SiameseNet, self).__init__()
        self.w, self.h, self.c = w, h, c

        # Encoder as ResNet like CNN with 4 blocks
        self.encoder = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=64, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)), Flatten()]
        )

    def call(self, support, query):
        pass

    def save(self, model_path):
        """
        Save encoder to the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder.save(model_path)

    def load(self, model_path):
        """
        Load encoder from the file.

        Args:
            model_path (str): path to the .h5 file.

        Returns: None

        """
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(model_path)


