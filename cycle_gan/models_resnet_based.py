import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
# https://github.com/keras-team/keras-io/blob/master/examples/generative/cyclegan.py

class ReflectionPadding2D(layers.Layer):
    """
    Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def downsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    use_bias=False,
):
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def residual_block(
    x,
    activation,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    use_bias=False,
):
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def Generator(
    img_height=256,
    img_width=256,
    filters=64,
    num_downsampling_blocks=2,
    num_residual_blocks=9,
    num_upsample_blocks=2,
    name=None,
):
    """
    Let c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k filters and stride 1. 
    dk denotes a 3×3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. 
    Reflection padding was used to reduce artifacts. 
    Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. 
    uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2.

    The architecture for the 6-resnet block generator for 128×128 images is as follows:
        c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3
    """
    kernel_initializer = tf.random_normal_initializer(0., 0.02)
    gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    img_input = layers.Input(
        shape=[img_height, img_width, 3], name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_initializer, use_bias=False)(
        x
    )
    x = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters,
                       activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


def Discriminator(
    img_height=256,
    img_width=256,
    filters=64,
    num_downsampling=3,
    name=None,
):
    """
    The discriminator model described in the paper takes 256×256 color images as input and 
    defines an explicit architecture that is used on all of the test problems. 
    The architecture uses blocks of Conv2D-InstanceNorm-LeakyReLU layers, with 4×4 filters and a 2×2 stride.

    Let Ck denote a 4×4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. 
    After the last layer, we apply a convolution to produce a 1-dimensional output. 
    We do not use InstanceNorm for the first C64 layer. We use leaky ReLUs with a slope of 0.2.

    The architecture for the discriminator is as follows:
        C64-C128-C256-C512
    This is referred to as a 3-layer PatchGAN in the CycleGAN and Pix2Pix nomenclature, 
    as excluding the first hidden layer, the model has three hidden layers that could be scaled up or down 
    to give different sized PatchGAN models.

    Not listed in the paper, the model also has a final hidden layer C512 with a 1×1 stride, 
    and an output layer C1, also with a 1×1 stride with a linear activation function. 
    Given the model is mostly used with 256×256 sized images as input, the size of the output feature map is 16×16. 
    """
    kernel_initializer = tf.random_normal_initializer(0., 0.02)

    img_input = layers.Input(
        shape=[img_height, img_width, 3], name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model
