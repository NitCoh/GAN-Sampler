'''
TODO LIST:
1. Understand the data, how to extract it and manipulate it to numpy arrays.
2. Build the train pipeline manually.
3. Define the architectures.
4. Debug on small data-set
5. Put it on Jupyter server.
6. Train.
'''
import tensorflow as tf
from scipy.io import arff
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from utils import handle_args_for_train
from gan_model import create_discriminator, create_gan, create_generator
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def pre_process_arff_data(filepath='german_credit.arff'):
    """
    Given ARFF file, transforming nominal data into one-hot encoding
    and histack the result to the right.
    Note:
    This function processing the whole data-set and assume it will fit into the memory.
    :param filepath: path to the arff file
    :return: numpy array represent the data, size: (num_samples, features)
    """
    data, meta = arff.loadarff(filepath)
    types = set(meta.types())
    print(f'Found types: {types}')
    x = None
    y = None
    cols = list(meta.names())
    for i, att in enumerate(cols):
        typo, classes = meta[att]

        if typo == 'nominal':
            res = preprocessing.label_binarize(data[att].astype(str), classes=classes)
        elif typo in ['real', 'numeric']:
            res = data[att]
            res = minmax_scale(res)  # min-max scale for all data
        else:
            res = data[att]

        if i == len(cols) - 1:  # assumes labels are last
            y = res
            continue

        if len(res.shape) == 1:
            res = np.expand_dims(res, axis=1)

        if x is None:
            x = res
        else:
            x = np.hstack([x, res])  # shape: (num_samples, x_right_shape + res_right_shape)

    return x, y


def split(x, y, percent=0.1):
    return train_test_split(x, y, test_size=percent, random_state=42)


def create_optimizer(lr):
    return tf.keras.optimizers.Adam(lr=lr, beta_1=0.5)


def generate_batch_indices(high, batch_size):
    import math
    sizes = [batch_size] * math.floor(high / batch_size)  # truncating the last batch
    return [np.random.randint(low=0, high=high, size=size) for size in sizes]


def plot_training_stats(d_loss, d_acc, g_loss):
    plt.plot(d_loss)
    plt.plot(g_loss)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Discriminator Loss', 'Generator Loss'], loc='upper left')
    plt.show()
    plt.plot(d_acc)
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(['Discriminator Accuracy'], loc='upper left')
    plt.show()


def analyse_gan(generator, discriminator, gan, x_test, y_test):
    num_samples, real_dim = x_test.shape
    gan_input_shape = 20

    noise = np.random.normal(0, 1, size=[100, gan_input_shape])
    generated_rows = generator.predict_on_batch(noise)

    subset = generated_rows[:10]
    distances = [np.linalg.norm(x_test[0] - x) for x in subset]
    print(f'Measuring euclidean distance from a real sample: {list(x_test[0])}')
    for gen_sample, dist in zip(list(subset), distances):
        print(f'Distance: {dist}, sample: {list(gen_sample)}')
    print('=' * 80)

    predicted_on_generated = discriminator.predict_on_batch(generated_rows)

    num_fooled = sum(predicted_on_generated > 0.5)

    subset_fooled = predicted_on_generated[predicted_on_generated > 0.5][:5]

    print(f'Num of fooled generated samples: {num_fooled}')

    print(f'Subset fooled: \n {list(subset_fooled)}')

    print()


def visualise_gan_output_during_training(model_path):
    pass


def part_a_train(args):
    import os
    import datetime
    import statistics
    tf.get_logger().setLevel('ERROR')

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f'./models/part_a/{time}'
    os.makedirs(dir)

    print(f'TF executing_eagerly: {tf.executing_eagerly()}')

    x, y = pre_process_arff_data()
    total_samples, real_dim = x.shape
    x_train, x_test, y_train, y_test = split(x, y)
    batch_size, epochs, lr, dim, optimizer = handle_args_for_train(args)
    gan_input_shape = 20

    # Creating the model
    generator = create_generator(batch_size, input_shape=gan_input_shape, dim=dim, real_dim=real_dim)
    discriminator = create_discriminator(batch_size, input_shape=real_dim, dim=dim)
    gan = create_gan(discriminator, generator, gan_input_shape)

    # compile the model
    # generator.compile(loss='binary_crossentropy', optimizer=create_optimizer(lr))
    discriminator.compile(loss='binary_crossentropy', optimizer=create_optimizer(lr), metrics=['accuracy'])
    gan.compile(loss='binary_crossentropy', optimizer=create_optimizer(lr))

    gan.summary()

    d_loss = []
    d_acc = []
    g_loss = []
    boost_generator = True

    for epoch in range(1, epochs + 1):
        batch_indices = generate_batch_indices(x_train.shape[0], batch_size)
        bar = tqdm(enumerate(batch_indices))
        d_epoch_loss = []
        d_epoch_acc = []
        g_epoch_loss = []
        if epoch % 50 == 0:
            boost_generator = not boost_generator
        # if epoch > 100:
        #     boost_generator = True

        for step, indices in bar:
            bar.set_description(f"Epoch: {epoch}")
            # generate noise
            noise = np.random.normal(0, 1, size=[batch_size, gan_input_shape])

            generated_rows = generator.predict(noise)  # (batch_size, real_dim)

            rows_batch = x_train[indices]

            x = np.vstack([rows_batch, generated_rows])

            # label start definition: 1 - real, 0 - fake
            y_disc = np.zeros(2 * batch_size)
            y_disc[:batch_size] = 0.9

            # if boost_generator:
            #     y_disc[:batch_size] = 0.8  # discourages the discriminator from being overconfident
            # else:
            #     y_disc[:batch_size] = 1.0

            # train discriminator
            discriminator.trainable = True
            d_cur_loss, d_cur_acc = discriminator.train_on_batch(x, y_disc)

            d_epoch_loss.append(d_cur_loss)
            d_epoch_acc.append(d_cur_acc)

            # generate noise
            noise = np.random.normal(0, 1, size=[batch_size, gan_input_shape])

            if boost_generator:
                y_gen = np.ones(batch_size)  # Flip labels to trick the discriminator
            else:
                y_gen = np.zeros(batch_size)

            # lock discriminator
            discriminator.trainable = False

            # train generator
            g_cur_loss = gan.train_on_batch(noise, y_gen)

            g_epoch_loss.append(g_cur_loss)

            bar.set_postfix(step=step, d_loss=d_cur_loss, d_acc=100 * d_cur_acc, g_loss=g_cur_loss)

        generator.save(dir + f'/epoch_{epoch}/generator/gen_model')
        discriminator.save(dir + f'/epoch_{epoch}/discriminator/disc_model')

        d_loss.append(statistics.mean(d_epoch_loss))
        d_acc.append(statistics.mean(d_epoch_acc))
        g_loss.append(statistics.mean(g_epoch_loss))

    plot_training_stats(d_loss, d_acc, g_loss)

    # tf.keras.utils.plot_model(gan, to_file=dir+f'/gan.png', show_shapes=True)
    # tf.keras.utils.plot_model(generator, to_file=dir + f'/gener.png', show_shapes=True)
    # tf.keras.utils.plot_model(discriminator, to_file=dir + f'/disc.png', show_shapes=True)

    analyse_gan(generator, discriminator, gan, x_test, y_test)

    print()

def train_classifier(args):
    pass


def part_b_train(args):
    pass


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(prog='main.py', description='GAN Sampler')
        parser.add_argument('-q', '--question', dest='question', action='store', type=int,
                            help='Which question to run', default=1, choices=[1, 2])
        parser.add_argument('-e', '--epochs', dest='epochs', action='store', type=int, default=500,
                            help='How many epochs the network will train')
        parser.add_argument('-opt', '--optimizer', dest='optimizer', metavar='optimizer', default='adam', type=str,
                            choices=['adam', 'sgd', 'rmsprop'],
                            help='The optimizer used for training')
        parser.add_argument('-d', dest='debug', action='store_true', help='use this flag to get debug prints')
        parser.add_argument('-b', '--batch_size', dest='batch_size', metavar='BATCH_SIZE', default=64, type=int,
                            help='the batch size used in train')
        parser.add_argument('--dim', dest='dim', metavar='DIMENSION', default=64, type=int,
                            help='The leading dimension in the GAN model')

        parser.add_argument('-lr', '--learning_rate', dest='lr', metavar='LEARNING_RATE', type=float,
                            default=1e-4, help='the learning rate used in training')
        parser.add_argument('-mp', '--model_path', dest='mp', metavar='MODEL_PATH', type=str,
                            default=None, help='the full path to the model you want to reconstruct it signals')
        args = parser.parse_args()

        return args


    questions = {1: part_a_train, 2: part_b_train}

    args = parse_args()
    q = args.question
    if q in questions:
        func = questions[q]
        func(args)
    else:
        print(f'Wrong choice of question to run')
