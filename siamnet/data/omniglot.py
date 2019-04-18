import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

WIDTH = 105
HEIGHT = 105

class DataLoader(object):
    def __init__(self, data, batch, n_classes, n_way):
        self.data = data
        self.batch = batch
        self.n_way = n_way
        self.n_classes = n_classes

    def get_next_episode(self):
        n_examples = 20
        support_batches = np.zeros([self.batch, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32)
        query_batches = np.zeros([self.batch, self.n_way**2, WIDTH, HEIGHT, 1], dtype=np.float32)
        labels_batches = np.zeros([self.batch, self.n_way**2])
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i_batch in range(self.batch):
            support = np.zeros([self.n_way, WIDTH, HEIGHT, 1])
            query = np.zeros([self.n_way, WIDTH, HEIGHT, 1])
            for i, i_class in enumerate(classes_ep):
                selected = np.random.permutation(n_examples)[:2]
                support[i] = self.data[i_class, selected[0]]
                query[i] = self.data[i_class, selected[1]]
            support_batch = np.take(support, [i // self.n_way for i in
                                                      range(self.n_way ** 2)], axis=0)
            query_batch = tf.tile(query, [self.n_way, 1, 1, 1])

            support_batches[i_batch] = support_batch
            query_batches[i_batch] = query_batch
            labels_batches[i_batch] = [i==j for i in range(self.n_way) for j in range(self.n_way)]

        support_batches = np.vstack(support_batches)
        query_batches = np.vstack(query_batches)
        labels_batches = labels_batches.reshape([-1, 1])

        return support_batches, query_batches, labels_batches


def class_names_to_paths(data_dir, class_names):
    """
    Return full paths to the directories containing classes of images.

    Args:
        data_dir (str): directory with dataset
        class_names (list): names of the classes in format alphabet/name/rotate

    Returns (list, list): list of paths to the classes,
    list of stings of rotations codes
    """
    d = []
    rots = []
    for class_name in class_names:
        alphabet, character, rot = class_name.split('/')
        image_dir = os.path.join(data_dir, 'data', alphabet, character)
        d.append(image_dir)
        rots.append(rot)
    return d, rots


def get_class_images_paths(dir_paths, rotates):
    """
    Return class names, paths to the corresponding images and rotations from
    the path of the classes' directories.

    Args:
        dir_paths (list): list of the class directories
        rotates (list): list of stings of rotation codes.

    Returns (list, list, list): list of class names, list of lists of paths to
    the images, list of rotation angles (0..240) as integers.

    """
    classes, img_paths, rotates_list = [], [], []
    for dir_path, rotate in zip(dir_paths, rotates):
        class_images = sorted(glob.glob(os.path.join(dir_path, '*.png')))

        classes.append(dir_path)
        img_paths.append(class_images)
        rotates_list.append(int(rotate[3:]))
    return classes, img_paths, rotates_list


def load_and_preprocess_image(img_path, rot):
    """
    Load and return preprocessed image.
    Args:
        img_path (str): path to the image on disk.
    Returns (Tensor): preprocessed image
    """
    img = Image.open(img_path).resize((WIDTH, HEIGHT)).rotate(rot)
    img = np.asarray(img)
    img = 1 - img
    return np.expand_dims(img, -1)


def load_class_images(n_support, n_query, img_paths, rot):
    """
    Given paths to the images of class, build support and query tf.Datasets.

    Args:
        n_support (int): number of images per support.
        n_query (int): number of images per query.
        img_paths (list): list of paths to the images for class.
        rot (int): rotation angle in degrees.

    Returns (tf.Dataset, tf.Dataset): support and query datasets.

    """
    # Shuffle indices of given images
    n_examples = img_paths.shape[0]
    example_inds = tf.range(n_examples)
    example_inds = tf.random.shuffle(example_inds)

    # Get indidces for support and query datasets
    support_inds = example_inds[:n_support]
    support_paths = tf.gather(img_paths, support_inds)
    query_inds = example_inds[n_support:]
    query_paths = tf.gather(img_paths, query_inds)

    # Build support dataset
    support_imgs = []
    for i in range(n_support):
        img = load_and_preprocess_image(support_paths[i])
        support_imgs.append(tf.expand_dims(img, 0))
    ds_support = tf.concat(support_imgs, axis=0)
    # Rotate dataset by given angle
    ds_support = tf.image.rot90(ds_support, k=rot//90)

    # Create query dataset
    query_imgs = []
    for i in range(n_query):
        img = load_and_preprocess_image(query_paths[i])
        query_imgs.append(tf.expand_dims(img, 0))
    ds_query = tf.concat(query_imgs, axis=0)
    # Rotate dataset by given angle
    ds_query = tf.image.rot90(ds_query, k=rot//90)

    return ds_support, ds_query


def load_omniglot(data_dir, config, splits):
    """
    Load omniglot dataset.

    Args:
        data_dir (str): path of the directory with 'splits', 'data' subdirs.
        config (dict): general dict with program settings.
        splits (list): list of strings 'train'|'val'|'test'

    Returns (dict): dictionary with keys as splits and values as tf.Dataset

    """
    split_dir = os.path.join(data_dir, 'splits', config['data.split'])
    ret = {}
    for split in splits:
        # n_way (number of classes per episode)
        if split in ['val', 'test']:
            n_way = config['data.test_way']
        else:
            n_way = config['data.train_way']

        batch = config['data.batch']

        # Get all class names
        class_names = []
        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))

        # Get class names, images paths and rotation angles per each class
        class_paths, rotates = class_names_to_paths(data_dir,
                                                    class_names)
        classes, img_paths, rotates = get_class_images_paths(
            class_paths,
            rotates)

        data = np.zeros([len(classes), len(img_paths[0]), WIDTH, HEIGHT, 1])
        for i_class in range(len(classes)):
            for i_img in range(len(img_paths[i_class])):
                data[i_class, i_img, :, :,:] = load_and_preprocess_image(
                            img_paths[i_class][i_img], rotates[i_class])

        data_loader = DataLoader(data,
                                 batch=batch,
                                 n_classes=len(classes),
                                 n_way=n_way)

        ret[split] = data_loader
    return ret
