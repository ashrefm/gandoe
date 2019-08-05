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


import math
import tensorflow as tf
from model.loss_fn import *
from model.models import encoder, decoder, discriminator


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


    # INPUTS
    # -------------------------------------------------------------------------

    img_channels = 3 if params.rgb else 1
    img_size = params.img_size
    z_size = params.z_size

    # Check if image size is power of 2
    if not math.ceil(math.log2(img_size)) == math.floor(math.log2(img_size)):
        raise ValueError("Image size in params files should be a power of 2. Found {}".format(img_size))


    images = tf.reshape(features, [-1, img_size, img_size, img_channels])
    assert images.shape[1:] == [img_size, img_size, img_channels], "{}"\
        .format(images.shape)
    # images = 2 * images - 1 (no need to adapt to generator's tanh)


    # MODEL
    # -------------------------------------------------------------------------

    with tf.variable_scope('model'):
    
        # Compute the encoded features of the original images 
        with tf.variable_scope('encoder1'):
            z = encoder(images, is_training, params)
            z_mean_norm = tf.reduce_mean(
                tf.norm(tf.reshape(z, [-1, params.z_size]), axis=1))

        # Generate fake images using the decoder
        with tf.variable_scope('decoder'):
            fakes = decoder(z, is_training, params, img_channels)

        # Compute the encoded features of the generated images 
        with tf.variable_scope('encoder2'):
            z_star = encoder(fakes, is_training, params)
            z_star_mean_norm = tf.reduce_mean(
                tf.norm(tf.reshape(z_star, [-1, params.z_size]), axis=1))

        # Feature extraction and discrimination of fake and real images
        with tf.variable_scope('discriminator'):
            features_real, logits_real = discriminator(
                images,
                is_training,
                params)
        
        with tf.variable_scope('discriminator', reuse=True):
            features_fake, logits_fake = discriminator(
                fakes,
                is_training,
                params,
                reuse=True)
 

    # LOSSES
    # -------------------------------------------------------------------------
    
    # contextual loss
    con_loss = context_loss(images, fakes)
    # feature matching loss
    fma_loss = l2_loss(features_real, features_fake)
    # adversarial loss
    adv_loss = bce_loss(logits_fake, tf.ones_like(logits_fake))
    # encoder loss
    enc_loss = l2_loss(z, z_star)

    # generator loss
    gen_loss = adv_loss


    if is_training:
        min_vs_max_enc_loss = tf.reduce_max(
            tf.reduce_mean(
                tf.reshape(tf.square(z - z_star), [-1, z_size]),
                axis=1)) 
        min_vs_max_con_loss = tf.reduce_max(
            tf.reduce_mean(
                tf.reshape(tf.abs(images - fakes), [-1, img_size*img_size*img_channels]),
                axis=1))
        min_vs_max_fma_loss = tf.reduce_max(
            tf.reduce_mean(
                tf.reshape(tf.square(features_real - features_fake), [-1, 4*4*(img_size*2)]),
                axis=1))
    else:
        min_vs_max_enc_loss = tf.reduce_min(
            tf.reduce_mean(
                tf.reshape(tf.square(z - z_star), [-1, z_size]),
                axis=1))
        min_vs_max_con_loss = tf.reduce_min(
            tf.reduce_mean(
                tf.reshape(tf.abs(images - fakes), [-1, img_size*img_size*img_channels]),
                axis=1))
        min_vs_max_fma_loss = tf.reduce_min(
            tf.reduce_mean(
                tf.reshape(tf.square(features_real - features_fake), [-1, 4*4*(img_size*2)]),
                axis=1))


    # discriminator loss
    real_loss = bce_loss(logits_real, tf.ones_like(logits_real))
    fake_loss = bce_loss(logits_fake, tf.zeros_like(logits_fake))
    dis_loss = real_loss + fake_loss

    if mode == tf.estimator.ModeKeys.PREDICT:

        # We will be returning individual losses for each observation
        predictions = {
            'z': z,
            'generated':fakes,
            'context_loss': tf.reduce_mean(
                tf.reshape(
                    tf.abs(images - fakes),
                    [-1, img_size * img_size * img_channels]
                ),
                axis=1),
            'encoder_loss': tf.reduce_mean(
                tf.reshape(tf.square(z - z_star), [-1, z_size]),
                axis=1)
        }

        export_outputs = {
            'output': tf.estimator.export.PredictedOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)


    # METRICS AND SUMMARIES
    # -------------------------------------------------------------------------
    with tf.variable_scope("metrics"):
        eval_metric_ops = {
            'z_mean_norm': tf.metrics.mean(z_mean_norm),
            'z_star_mean_norm': tf.metrics.mean(z_star_mean_norm),
            'context_loss': tf.metrics.mean(con_loss),
            'min_vs_max_context_loss': tf.metrics.mean(min_vs_max_con_loss),
            'feature_matching_loss': tf.metrics.mean(fma_loss),
            'min_vs_max_fma_loss': tf.metrics.mean(min_vs_max_fma_loss),
            'adversarial_loss': tf.metrics.mean(adv_loss),
            'encoder_loss': tf.metrics.mean(enc_loss),
            'min_vs_max_encoder_loss': tf.metrics.mean(min_vs_max_enc_loss),
            'generator_loss': tf.metrics.mean(gen_loss),
            'real_loss': tf.metrics.mean(real_loss),
            'fake_loss': tf.metrics.mean(fake_loss),
            'discriminator_loss': tf.metrics.mean(dis_loss)
        }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=gen_loss+dis_loss,
            eval_metric_ops=eval_metric_ops)

    # Scalar summaries for training
    tf.summary.scalar("z_mean_norm", z_mean_norm)
    tf.summary.scalar("z_star_mean_norm", z_star_mean_norm)
    tf.summary.scalar("context_loss", con_loss)
    tf.summary.scalar("min_vs_max_context_loss", min_vs_max_con_loss)
    tf.summary.scalar("feature_matching_loss", fma_loss)
    tf.summary.scalar("min_vs_max_fma_loss", min_vs_max_fma_loss)
    tf.summary.scalar("adversarial_loss", adv_loss)
    tf.summary.scalar("encoder_loss", enc_loss)
    tf.summary.scalar("min_vs_max_encoder_loss", min_vs_max_enc_loss)
    tf.summary.scalar("generator_loss", gen_loss)
    tf.summary.scalar("real_loss", real_loss)
    tf.summary.scalar("fake_loss", fake_loss)
    tf.summary.scalar("discriminator_loss", dis_loss)

    # Image summaries for training
    tf.summary.image('real_images', images, max_outputs=8)
    tf.summary.image('fake_images', fakes, max_outputs=8)


    # OPTIMIZATION
    # -------------------------------------------------------------------------

    # Define the optimizer based on choice in the configuration file
    optimizers = {'adam': tf.train.AdamOptimizer, 
                  'adagrad': tf.train.AdagradOptimizer,
                  'adadelta': tf.train.AdadeltaOptimizer,
                  'rmsprop': tf.train.RMSPropOptimizer,
                  'gradient_descent': tf.train.GradientDescentOptimizer}

    if params.optimizer in list(optimizers.keys()):
        gen_optimizer = optimizers[params.optimizer](params.learning_rate)
        enc_optimizer = optimizers[params.optimizer](params.learning_rate)
        dis_optimizer = optimizers[params.optimizer](params.learning_rate)
    else:
        raise ValueError(
            "Optimizer not recognized: {}\nShould be in the list {}".\
            format(params.optimizer, list(optimizers.keys())))

    tf.logging.info("Current optimizer: {}".format(params.optimizer))

    # Define training step that minimizes the loss with the chosen optimizer
    global_step = tf.train.get_global_step()

    # Connect the different optimizers
    train_op = tf.group(
        gen_optimizer.minimize(gen_loss, global_step=global_step),
        enc_optimizer.minimize(enc_loss, global_step=global_step),
        dis_optimizer.minimize(dis_loss, global_step=global_step))
    
    return tf.estimator.EstimatorSpec(
        mode,
        loss=gen_loss+enc_loss+dis_loss,
        train_op=train_op)
