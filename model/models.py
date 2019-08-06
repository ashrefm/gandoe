#  Copyright 2019 Mohamed-Achref MAIZA. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Model architectures"""


import tensorflow as tf


def _batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_training=True):
    """Tensorflow based batch normalization."""
    
    return tf.contrib.layers.batch_norm(x,
        decay=momentum,
        updates_collections=None,
        epsilon=epsilon,
        scale=True,
        is_training=is_training,
        scope=name)


def encoder(images, is_training, params):
    """Sub-network that computes the latent vector.

    Args:
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) dictionary of hyperparameters
        is_training: (bool) flag to indicate training mode

    Returns:
        z: (tf.Tensor) batch of latent vector output by the encoder 
    """

    x = images
    # 128 x 128 x num_channels

    if params.img_size != 128:
        raise ValueError("Image size should be equal to 128 if you want to use this model architecture.")
 
    # Init variables
    channels = params.initial_filters
    size = params.img_size
    conv_idx = 1

    # Encoding
    while size > 4:

        x = tf.layers.conv2d(x,
            channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False,
            name='conv2d_'+str(conv_idx))

        x = _batch_norm(x,
            name='batch_norm_'+str(size)+'_'+str(size)+'_'+str(channels),
            is_training=is_training)

        x = tf.nn.relu(x, name='relu_'+str(conv_idx))
        size = size // 2
        channels = channels * 2
        conv_idx +=1
    # 4 x 4 x channels
    
    z = tf.layers.conv2d(x,
        params.z_size, 
        kernel_size=4,
        padding='valid',
        use_bias=False,
        name='conv2d_'+str(conv_idx+1))
    # 1 x 1 x z_size (latent vector)

    return z


def decoder(z, is_training, params, img_channels):
    """Sub-network that generates a fake image.

    Args:
        z: (tf.Tensor) latent vector of shape (1 x 1 x params.z_size)
        is_training: (bool) flag to indicate training mode
        params: (Params) dictionary of hyperparameters
        img_channels: (int) number of channels in the reconstructed image

    Returns:
        decoded: (tf.Tensor) batch of fake images output by the decoder
    """

    x = z
    # 1 x 1 x params.z_size

    # Find number of filters for the fist convolution in decoder
    # Should be the same as the second last in encoder
    channels = params.initial_filters
    size = params.img_size
    while size > 4:
        size = size // 2
        channels = channels * 2

    # Init variables
    size = 1
    conv_idx = 1

    # Decoding
    while size < params.img_size // 2:

        x = tf.keras.layers.Conv2DTranspose(channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False,
            name='conv2dT_'+str(conv_idx))(x)

        x = _batch_norm(x,
            name='batch_norm_'+str(size)+'_'+str(channels),
            is_training=is_training)

        x = tf.nn.relu(x, name='relu_'+str(conv_idx))
        size = size * 2
        channels = channels // 2
        conv_idx +=1

    x = tf.keras.layers.Conv2DTranspose(img_channels,    
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=False,
        name='conv2dT_'+str(conv_idx+1))(x)
    decoded = tf.nn.tanh(x)
    # 128 x 128 x img_channels (final layer)

    return decoded


def discriminator(inputs, is_training, params):
    """Sub-network that classifies input images as real or fake.

    Args:
        inputs: (tf.Tensor) batch of input images
        is_training: (bool) flag to indicate training mode
        params: (Params) dictionary of hyperparameters

    Returns:
        features: (tf.Tensor) batch of extracted features in second last conv layer
        logits: (tf.Tensor) batch of logits for real
    """

    x = inputs
    # 128 x 128 x 3

    # Init variables
    channels = params.initial_filters
    size = params.img_size
    conv_idx = 1
        
    # Discriminate
    while size > 4:

        x = tf.layers.conv2d(x,
            channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False,
            name='conv2d_'+str(conv_idx))
                
        x = _batch_norm(x,
            name='batch_norm_'+str(size)+'_'+str(size)+'_'+str(channels),
            is_training=is_training)

        x = tf.nn.leaky_relu(x, alpha=0.2, name='leaky_relu_'+str(conv_idx))
        size = size // 2
        channels = channels * 2
        conv_idx +=1

    features = x
    # 4 x 4 x channels

    x = tf.layers.conv2d(x,
        channels,
        kernel_size=4,
        strides=1,
        padding='valid',
        use_bias=False,
        name='conv2d_'+(str(conv_idx+1)))

    x = tf.layers.flatten(x)
    logits = tf.layers.dense(x, 1)

    return features, logits
