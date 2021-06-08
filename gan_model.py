import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, ReLU

leaky_alpha = 0.4

def create_discriminator(batch_size, input_shape, dim):
    input = Input(shape=input_shape, batch_size=batch_size)
    x = Dense(dim * 4, activation=LeakyReLU(alpha=leaky_alpha))(input)
    # x = Dropout(0.4)(x)
    x = Dense(dim * 3, activation=LeakyReLU(alpha=leaky_alpha))(x)
    # x = Dropout(0.4)(x)
    x = Dense(dim * 2, activation=LeakyReLU(alpha=leaky_alpha))(x)
    # x = Dropout(0.4)(x)
    x = Dense(dim, activation=LeakyReLU(alpha=leaky_alpha))(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input, outputs=x)

def create_generator(batch_size, input_shape,dim, real_dim):
    input = Input(shape=input_shape, batch_size=batch_size)
    x = Dense(dim, activation=LeakyReLU(alpha=leaky_alpha))(input)
    x = Dense(dim * 2, activation=LeakyReLU(alpha=leaky_alpha))(x)
    x = Dense(dim * 4, activation=LeakyReLU(alpha=leaky_alpha))(x)
    x = Dense(real_dim, activation=ReLU(max_value=1))(x)
    return Model(inputs=input, outputs=x)


def create_gan(discriminator, generator, data_shape):
    gan_input = Input(shape=(data_shape, ))
    x = generator(gan_input)
    gan_output = discriminator(x)
    return Model(inputs=gan_input, outputs=gan_output)

