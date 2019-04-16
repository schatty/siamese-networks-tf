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

        self.dense = tf.keras.Sequential([
            Dense(1, activation='sigmoid')
        ])

    @tf.function
    def call(self, support, query):
        n_class = support.shape[0]
        y = np.arange(n_class)[:, np.newaxis]
        y_onehot = tf.cast(tf.one_hot(y, n_class), tf.float32)

        print("Support & query shapes before: ", support.shape, query.shape)

        support = tf.gather(support, [i//n_class for i in range(n_class**2)])
        query = tf.tile(query, [n_class, 1, 1, 1])

        labels = tf.cast([i==j for i in range(n_class) for j in range(n_class)], tf.float32)
        labels = tf.reshape(labels, [-1, 1])

        print("Support & query shapes after: ", support.shape, query.shape)

        cat = tf.concat([support, query], axis=0)
        z = self.encoder(cat)
        print("z shape: ", z.shape)

        z_support = z[:n_class**2]
        # Prototypes are means of n_support examples
        z_query = z[n_class**2:]

        print("Z support: ", z_support.shape)
        print("Z query: ", z_query.shape)

        l1_dist = tf.abs(z_support - z_query)
        print("l1_dist: ", l1_dist.shape)

        score = self.dense(l1_dist)

        loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_true=labels, y_pred=score))
        print("Score shape: ", score.shape)
        labels_pred = tf.cast(tf.argmax(tf.reshape(score, [-1, n_class]), -1), tf.float32)
        labels_true = tf.range(0, n_class, dtype=tf.float32)
        eq = tf.cast(tf.equal(labels_pred, labels_true), tf.float32)
        acc = tf.reduce_mean(eq)

        return loss, acc

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


