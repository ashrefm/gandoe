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

"""Define loss functions."""


import tensorflow as tf


def l1_loss(tensor1, tensor2):
    """Computes the average l1 distance between two tensors."""

    l1_dist = tf.reduce_mean(tf.abs(tensor1 - tensor2))

    return l1_dist
    

def l2_loss(tensor1, tensor2):
    """Computes the average l2 distance between two tensors."""

    l2_dist = tf.reduce_mean(tf.square(tensor1 - tensor2))

    return l2_dist


def bce_loss(y_pred, y_true):
    """Computes the average binary cross-entropy."""

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))


def context_loss(inputs, outputs):
    """Computes the average reconstruction loss using the l1 distance.

    Args:
        inputs: (tf.Tensor) input batch of images (img_size, img_size, img_channels)
        outputs: (tf.Ttensor) output batch of images (img_size, img_size , img_channels)

    Returns:
        l1_dist: (tf.float32) scalar tensor of context loss
    """

    l1_dist = tf.reduce_mean(tf.abs(inputs - outputs))

    return l1_dist


def adversarial_loss(features_real, features_fake):
    """Computes the real and fakes features matching inside discriminator.

    Args:
        features_real: (tf.Tensor) input batch of features (w, h, c)
        features_fake: (tf.Ttensor) output batch of images (w, h, c)

    Returns:
        l2_loss: (tf.float32) scalar tensor of adversarial loss
    """

    l2_dist = tf.reduce_mean(tf.square(features_real - features_fake))

    return l2_dist
