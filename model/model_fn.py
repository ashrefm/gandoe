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

"""Define the model."""


import tensorflow as tf
from model.loss import context_loss
from model import models, loss
from model.models import encoder, decoder


def model_fn(features, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    img_channels = 3 if params.rgb else 1
    img_size = params.img_size

    images = tf.reshape(features, [-1, img_size, img_size, img_channels])
    assert images.shape[1:] == [img_size, img_size, img_channels], "{}".format(images.shape)

    # Generate fake images using the generator (encoder + decoder)
    # ------------------------------------------------------------
    subnet_name = 'encoder1'
    with tf.variable_scope(subnet_name):
        z = encoder(is_training, images, params, subnet_name)
    z_mean_norm = tf.reduce_mean(tf.norm(z, axis=1))

    subnet_name = 'decoder'
    with tf.variable_scope(subnet_name):
        decoded = decoder(is_training, z, params, img_channels)

    # Compute the loss metrcis
    # ------------------------------------------------------------
    loss1 = context_loss(images, decoded)
    loss = loss1


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'z': z, 'decoded':decoded, 'context_loss': loss1}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # METRICS AND SUMMARIES
    # -----------------------------------------------------------
    with tf.variable_scope("metrics"):
        eval_metric_ops = {
            'z_mean_norm': tf.metrics.mean(z_mean_norm),
            'context_loss': tf.metrics.mean(loss1)
        }


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar("z_mean_norm", z_mean_norm)
    tf.summary.scalar("context_loss", loss1)
    tf.summary.scalar("loss", loss)


    tf.summary.image('train_image', images, max_outputs=8)
    tf.summary.image('generator_image', decoded, max_outputs=8)

    # Define the optimizer based on choice in the configuration file
    optimizers = {'adam': tf.train.AdamOptimizer, 
                  'adagrad': tf.train.AdagradOptimizer,
                  'adadelta': tf.train.AdadeltaOptimizer,
                  'rmsprop': tf.train.RMSPropOptimizer,
                  'gradient_descent': tf.train.GradientDescentOptimizer}

    if params.optimizer in list(optimizers.keys()):
        optimizer = optimizers[params.optimizer](params.learning_rate)
    else:
        raise ValueError("Optimizer not recognized: {}\nShould be in the list {}".format(params.optimizer, list(optimizers.keys())))

    tf.logging.info("Current optimizer: {}".format(params.optimizer))


    # Define training step that minimizes the loss with the chosen optimizer
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)
    

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
