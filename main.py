import os
import time
import numpy as np
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)

from omniglot import load_omniglot as load
from train_engine import TrainEngine
from siamese import SiameseNet


def train(config):
    np.random.seed(2019)
    tf.random.set_seed(2019)

    # Create folder for model
    model_dir = config['model.save_path'][:config['model.save_path'].rfind('/')]
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_dir = f"data/{config['data.dataset']}"
    ret = load(data_dir, config, ['train', 'val'])
    train_loader = ret['train']
    val_loader = ret['val']

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = SiameseNet(w, h, c)
    optimizer = tf.keras.optimizers.Adam(config['train.lr'])


    # Metrics to gather
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    val_losses = []

    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    def train_step(loss_func, support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    train_engine = TrainEngine()

    # Set hooks on training engine
    def on_start(state):
        print("Training started.")

    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")

    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()

    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'
        print(
            template.format(epoch + 1, train_loss.result(),
                            train_acc.result() * 100,
                            val_loss.result(),
                            val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: ", cur_loss)
            state['best_val_loss'] = cur_loss
            model.save(config['model.save_path'])
        val_losses.append(cur_loss)

        # Early stopping
        patience = config['train.patience']
        if len(val_losses) > patience \
                and max(val_losses[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True

    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        if state['total_episode'] % 20 == 0:
            print(f"Episode {state['total_episode']}")
        support, query = state['sample']
        loss_func = state['loss_func']
        train_step(loss_func, support, query)

    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        # Validation
        val_loader = state['val_loader']
        loss_func = state['loss_func']
        for i_episode in range(config['data.episodes']):
            support, query = val_loader.get_next_episode()
            val_step(loss_func, support, query)
    train_engine.hooks['on_end_episode'] = on_end_episode

    time_start = time.time()
    with tf.device(device_name):
        train_engine.train(
            loss_func=loss,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['train.epochs'],
            n_episodes=config['data.episodes'])
    time_end = time.time()

    elapsed = time_end - time_start
    h, min = elapsed // 3600, elapsed % 3600 // 60
    sec = elapsed - min * 60
    print(f"Training took: {h} h {min} min {sec} sec")


def eval(config):
    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    # Setup training operations
    w, h, c = list(map(int, config['model.x_dim'].split(',')))
    model = SiameseNet(w, h, c)

    # Metrics to gather
    val_loss = tf.metrics.Mean(name='val_loss')
    val_acc = tf.metrics.Mean(name='val_accuracy')

    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    with tf.device(device_name):
        pass


if __name__ == "__main__":
    config = {
        'data.train_way': 2,
        'data.test_way': 2,
        'data.dataset': 'omniglot',
        'data.split': 'vinyals',
        'data.episodes': 100,
        'data.cuda': 0,
        'data.gpu': 0,

        'train.epochs': 15,
        'train.lr': 0.001,
        'train.patience': 100,

        'model.x_dim': '28,28,1',
        'model.save_path': 'results/models/test.h5'
    }
    train(config)