import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model


class SiameseNet(Model):
    """ Implementation for Siamese networks. """

    def __init__(self, w, h, c, way=2):
        super(SiameseNet, self).__init__()
        self.w, self.h, self.c = w, h, c
        self.way = way

        # CNN-encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=10, padding='valid'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=128, kernel_size=7, padding='valid'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=128, kernel_size=4, padding='valid'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=128, kernel_size=4, padding='valid'),
            BatchNormalization(),
            ReLU(),

            Flatten()]
        )

        # Final dense layer after L1 norm
        self.dense = tf.keras.Sequential([
            Dense(1, activation='sigmoid')
        ])

    @tf.function
    def call(self, support, query, labels):
        # Number of elements in batch
        n = support.shape[0]

        # Concatenate and forward through encoder
        cat = tf.concat([support, query], axis=0)
        z = self.encoder(cat)

        # Spilt back to support and query parts
        z_support = z[:n]
        z_query = z[n:]

        # L1-norm
        l1_dist = tf.abs(z_support - z_query)

        # Final dense layer
        score = self.dense(l1_dist)

        # Loss and accuracy
        loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_true=labels, y_pred=score))
        labels_pred = tf.cast(tf.argmax(tf.reshape(score, [-1, self.way]), -1), tf.float32)
        labels_true = tf.tile(tf.range(0, self.way, dtype=tf.float32), [n//self.way**2])
        eq = tf.cast(tf.equal(labels_pred, labels_true), tf.float32)
        acc = tf.reduce_mean(eq)

        return loss, acc

    def save(self, save_dir):
        """
        Save encoder to the file.

        Args:
            save_dir (str): path to the .h5 file.

        Returns: None

        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        encoder_path = os.path.join(save_dir, 'encoder.h5')
        self.encoder.save(encoder_path)
        dense_path = os.path.join(save_dir, 'dense.h5')
        self.dense.save(dense_path)

    def load(self, dir):
        """
        Load encoder from the file.

        Args:
            dir (str): path to the .h5 file.

        Returns: None

        """
        encoder_path = os.path.join(dir, 'encoder.h5')
        self.encoder(tf.zeros([1, self.w, self.h, self.c]))
        self.encoder.load_weights(encoder_path)
        dense_path = os.path.join(dir, 'dense.h5')
        self.dense(tf.zeros([1, 4608]))
        self.dense.load_weights(dense_path)


