import unittest
from scripts import train

gpu_num = 0


class TestsOmniglot(unittest.TestCase):

    def test_2_way_batch_4(self):
        config = {
            'data.dataset_path': '/home/igor/dl/siamese-networks-tf/data/omniglot',
            'data.dataset': 'omniglot',
            'data.train_way': 2,
            'data.test_way': 2,
            'data.split': 'vinyals',
            'data.batch': 4,
            'data.episodes': 2,
            'data.cuda': 1,
            'data.gpu': gpu_num,
            'train.epochs': 1,
            'train.lr': 0.001,
            'train.patience': 100,
            'train.tb_dir': 'results/logs/gradient_tape/',
            'train.log_dir': 'results/logs',
            'train.restore': 0,
            'model.x_dim': '105,105,1',
            'model.save_dir': 'results/models/omniglot'
        }
        train(config)

        config['train.restore'] = 1
        train(config)

    def test_2_way_batch_16(self):
        config = {
            'data.dataset_path': '/home/igor/dl/siamese-networks-tf/data/omniglot',
            'data.dataset': 'omniglot',
            'data.train_way': 2,
            'data.test_way': 2,
            'data.split': 'vinyals',
            'data.batch': 16,
            'data.episodes': 2,
            'data.cuda': 1,
            'data.gpu': gpu_num,
            'train.epochs': 1,
            'train.lr': 0.001,
            'train.patience': 100,
            'train.tb_dir': 'results/logs/gradient_tape/',
            'train.log_dir': 'results/logs',
            'train.restore': 0,
            'model.x_dim': '105,105,1',
            'model.save_dir': 'results/models/omniglot'
        }
        train(config)

        config['train.restore'] = 1
        train(config)

    def test_2_way_batch_1(self):
        config = {
            'data.dataset_path': '/home/igor/dl/siamese-networks-tf/data/omniglot',
            'data.dataset': 'omniglot',
            'data.train_way': 2,
            'data.test_way': 2,
            'data.split': 'vinyals',
            'data.batch': 1,
            'data.episodes': 2,
            'data.cuda': 1,
            'data.gpu': gpu_num,
            'train.epochs': 1,
            'train.lr': 0.001,
            'train.patience': 100,
            'train.tb_dir': 'results/logs/gradient_tape/',
            'train.log_dir': 'results/logs',
            'train.restore': 0,
            'model.x_dim': '105,105,1',
            'model.save_dir': 'results/models/omniglot'
        }
        train(config)

        config['train.restore'] = 1
        train(config)