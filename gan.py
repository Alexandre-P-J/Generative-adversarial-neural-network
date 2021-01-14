import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # hide tensorflow info warnings
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np


# Dataset
batch_size = 1024  # size of a minibatch
# Training
epochs = 50
noise_dim = 100
# Checkpoints
checkpoint_dir = './training_checkpoints'


def get_dataset():
    # For this experiment i dont need the labels neither the train test split so both splits are merged
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    images = np.concatenate((train_images, test_images))
    images_count = images.shape[0]

    images = images.reshape(images_count, 28, 28, 1).astype('float32')
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    images_dataset = tf.data.Dataset.from_tensor_slices(
        images).shuffle(images_count).batch(batch_size)
    return images_dataset


# Configures and tries to load last checkpoint
def get_checkpoint_manager(generator_optimizer, discriminator_optimizer, generator, discriminator):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=3)
    return manager


def get_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def get_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(
        generated_output), generated_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(generated_output), generated_output)


@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        discr_loss = discriminator_loss(real_output, generated_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        discr_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    generator = get_generator_model()
    discriminator = get_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # Create and try to load last checkpoint
    checkpoint_manager = get_checkpoint_manager(
        generator_optimizer, discriminator_optimizer, generator, discriminator)
    checkpoint_manager.restore_or_initialize()

    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator,
                       generator_optimizer, discriminator_optimizer)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_manager.save()

        print(f"Time for epoch {epoch + 1} is {time.time() - start:.0f}s")


def play():
    generator = get_generator_model()
    discriminator = get_discriminator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # Create and try to load last checkpoint
    checkpoint_manager = get_checkpoint_manager(
        generator_optimizer, discriminator_optimizer, generator, discriminator)
    checkpoint_manager.restore_or_initialize()

    # Slideshow
    img = None
    fig = plt.figure()
    fig_id = fig.number
    while plt.fignum_exists(fig_id):
        # May be better to generate more than one image at a time
        noise = tf.random.normal([1, noise_dim])
        image = generator(noise, training=False)[0, :, :, 0]
        if img is None:
            img = plt.imshow(image, cmap='gray')
        else:
            img.set_data(image)
        plt.draw()
        plt.pause(1)


def main():
    dataset = get_dataset()
    train(dataset, epochs)
    #play()


if __name__ == "__main__":
    main()
