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

"""Train the model"""


import argparse
import os
import time
import timeit

import tensorflow as tf

from model.input_fn import read_dataset
from model.model_fn import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/dataset',
                    help="Directory containing the dataset")


if __name__ == '__main__':

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    train_steps = params.train_examples // params.batch_size
    checkpoint_steps = train_steps // params.checkpoints
    summary_steps = checkpoint_steps // 5

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(
        tf_random_seed=22,
        model_dir=args.model_dir,
        save_checkpoints_steps=checkpoint_steps,
        save_summary_steps=summary_steps
    )

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Specification for training the model
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(
            os.path.join(args.data_dir, 'train'),
            params,
            mode=tf.estimator.ModeKeys.TRAIN
        ),
        max_steps=train_steps+1
    )

    # Specification for evaluating the model
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(
            os.path.join(args.data_dir, 'dev', 'KO'),
            params,
            mode=tf.estimator.ModeKeys.EVAL
        ),
        steps=checkpoint_steps,
        start_delay_secs=1, # Start evaluation after 1 second if checkpoint available
        throttle_secs=params.eval_interval # evaluate if new checkpoints available and last evaluation at least eval_interval seconds ago
    )

    # Train and evaluate the model
    tf.logging.info("Starting training for {} step(s).".format(train_steps))
    start = time.clock()
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    tf.logging.info("Training took %s seconds." % (time.clock() - start))
