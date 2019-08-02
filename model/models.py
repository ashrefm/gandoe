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


def encoder(is_training, images, params, name):
    """Compute outputs of the model (embeddings for triplet loss).
    Adding L2-norm layer to miniception_v2
    (maybe add a learnable scaling parameter alpha, see paper L2-constraint softmax)

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters
        name: (String) name of the encoder

    Returns:
        z: (tf.Tensor) output of the encoder (latent vector)
    """

    x = images
    # 128 x 128 x num_channels

    if params.img_size != 128:
        raise ValueError("Image size should be equal to 128 if you want to use this model architecture.")
 
    channels = params.initial_filters
    size = params.img_size

    while size > 4:

        x = tf.layers.conv2d(x,
            channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False)

        x = _batch_norm(x,
            name=name+'_bn_'+str(size)+'_'+str(size)+'_'+str(channels),
            is_training=is_training)

        x = tf.nn.relu(x)
        size = size // 2
        channel = channels * 2
    
    z = tf.layers.conv2d(x, params.z_size, 4, padding='valid', use_bias=False)
    # 1 x 1 x z_size (latent vecotr)

    return z


def decoder(is_training, z, params, img_channels):

    channels = params.initial_filters
    size = params.img_size
    while size > 4:
        size = size // 2
        channels = channels * 2

    size = 1
    x = z
    while size < params.img_size // 2:

        x = tf.keras.layers.Conv2DTranspose(channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False)(x)

        x = _batch_norm(x,
            name='decoder_bn_'+str(size)+'_'+str(channels),
            is_training=is_training)

        x = tf.nn.relu(x)
        size = size * 2
        channels = channels // 2

    x = tf.keras.layers.Conv2DTranspose(img_channels,    
        kernel_size=4,
        strides=2,
        padding='same',
        use_bias=False)(x)
    decoded = tf.nn.tanh(x)
    # 128 x 128 x img_channels (final layer)

    return decoded

