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

"""Create the input data pipeline using `tf.data`"""


import glob
import os
import tensorflow as tf


def _parse_example(filename, img_size, channels):
    # Read an image from a file
    # Decode it into a dense vector
    # Resize it to fixed shape
    # Reshape it to 1 dimensonal tensor
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    image_resized = tf.image.resize_images(image_decoded, [img_size, img_size])
    features = tf.reshape(image_resized, [img_size*img_size*channels])
    features_normalized = features / 255.0

    return features_normalized


def _check_dir(dataset_dir, params):
    """Extracts valid images from dataset folder.

    Args:
        dataset_dir: directory containing the train, validation or test images
        params: contains hyperparameters of the model (ex: `params.learning_rate`)
    
    Returns:
        image_list: (list) containing all valid images
    """

    if not os.path.exists(dataset_dir):
        raise ValueError('Dataset path is not correct {}'.format(dataset_dir))

    # get all images from each class folder
    image_list = glob.glob(dataset_dir+"/*."+params.img_type)
    if len(image_list) == 0:
        raise ValueError('No valid images were found in %s' %dataset_dir)
    
    return image_list


def create_dataset(dataset_dir, params):
    """Load and parse dataset.

    Args:
        dataset_dir: directory containing the train, validation or test images
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        dataset: (tf.data.Dataset) containing filenames
    """

    filenames = []

    # get all images from each class folder
    image_list = _check_dir(dataset_dir, params)

    # add class images to filenames
    filenames = filenames + image_list

    print(filenames[:10])

    # Create the dataset using a python generator to save memory and reduce size of events dump file
    def generator():
        for sample in filenames:
            yield sample
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=tf.string,
                                             output_shapes=tf.TensorShape([]))

    return dataset


def read_dataset(dataset_dir, params, mode):
    """Train input function.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    
    Returns:
        input_fn: (function) callable input function that generates a dataset
    """

    def _input_fn():
        dataset = create_dataset(dataset_dir, params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # Indefinetely
            dataset = dataset.shuffle(10*params.batch_size, seed=22, reshuffle_each_iteration=True)  # whole dataset into the buffer
        else:
            num_epochs = 1 # end of input after one epoch

        dataset = dataset.repeat(num_epochs)
        channels = 3 if params.rgb else 1
        dataset = dataset.map(
            lambda filename: _parse_example(filename, params.img_size, channels),
            num_parallel_calls=5)
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve

        return dataset

    return _input_fn
