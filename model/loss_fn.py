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


def bce_loss_with_logits(y_hat, y):
    """Computes the average binary cross-entropy based on logits."""

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                                  logits=y_hat))


def bce_loss(y_hat, y):
    """Computes the average binary cross-entropy based on probabilities."""
    
    bce_loss = - y * tf.log(tf.maximum(y_hat, 1e-16)) - (1-y) * tf.log(tf.maximum(1-y_hat, 1e-16))
    bce_loss = tf.reduce_mean(bce_loss)

    return bce_loss 


def context_loss(inputs, outputs, margin=0):
    """Computes the average reconstruction loss using the l1 distance.

    Args:
        inputs: (tf.Tensor) input batch of images (img_size, img_size, img_channels)
        outputs: (tf.Ttensor) output batch of images (img_size, img_size , img_channels)
        margin: (float) minimize distance to a certain percentage of margin

    Returns:
        l1_dist: (tf.float32) scalar tensor of context loss
    """

    l1_dist = tf.reduce_mean(tf.abs(inputs - outputs) + margin)

    return l1_dist