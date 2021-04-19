import tensorflow as tf
from tensorflow import keras


def discriminator_loss(real, generated):
    """
    The discriminator loss function below compares real images to a matrix of 1s and 
    fake images to a matrix of 0s. 
    The perfect discriminator will output all 1s for real images and all 0s for fake images. 
    The discriminator loss outputs the average of the real and generated loss.
    """

    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    """
    The generator wants to fool the discriminator into thinking the generated image is real. 
    The perfect generator will have the discriminator output only 1s. 
    Thus, it compares the generated image to a matrix of 1s to find the loss.
    """

    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)


def discriminator_loss_lsgan(real, generated):
    """
    https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L775
    """

    real_loss = tf.reduce_mean(
        tf.math.squared_difference(tf.ones_like(real), real))
    generated_loss = tf.reduce_mean(
        tf.math.squared_difference(tf.zeros_like(generated), generated))

    total_disc_loss = (real_loss + generated_loss)
    return total_disc_loss * 0.5


def generator_loss_lsgan(generated):
    """
    https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/losses/losses_impl.py#L775
    """

    return tf.reduce_mean(tf.math.squared_difference(tf.ones_like(generated), generated))


def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    """
    We want our original photo and the twice transformed photo to be similar to one another. 
    Thus, we can calculate the cycle consistency loss as finding the average of their difference.
    e.g.
    generator_monet: photo -> monet
    generator_photo: monet -> photo

    generated_monet = generator_monet(real_photo) 
    cycled_photo = generator_photo(generated_monet)
    calc_cycle_loss(real_photo, cycled_photo)
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image, LAMBDA):
    """
    Identity loss says that, if you fed image to generator, it should 
    yield the real image or something close to it.
    If you run the zebra-to-horse model on a horse or the horse-to-zebra model on a zebra, 
    it should not modify the image much since the image already contains the target class.
    e.g.
    generator_monet: photo -> monet
    generator_photo: monet -> photo

    same_monet = generator_monet(real_monet) 
    identity_loss(real_monet, same_monet)
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss
