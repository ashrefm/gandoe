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
import numpy as np


def context_loss(inputs, outputs):
    """Computes the average reconstruction loss using the norm 1

    Args:
        inputs: (tf.Tensor) input batch of images (img_size, img_size, img_channels)
        outputs: (tf.Ttensor) output batch of images (img_size, img_size , img_channels)

    Returns:
        l1_loss: tf.estimator.EstimatorSpec object
    """

    l1_loss = tf.reduce_mean(tf.abs(inputs - outputs))

    return l1_loss







def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def _rank1_accuracy(pairwise_dist, labels):
    """Return the rank-1 accuracy given a distance matrix and labels.

    Args:
        pairwise_dist: 2D matrix of distances between all the embeddings.
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # First, we need to get a 2D mask for every valid anchor-positive (they should have same label and be distinct)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)
    # We also need to get a 2D mask for every anchor-negative (distinct labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)
    # Create a 2D mask in which all diagonal elements are set to infinite and the remaining is set to 0
    batch_size = tf.shape(labels)[0]
    diag_to_inf_mask = 1e16 * tf.eye(num_rows=batch_size)
    # Get the minimum distance for each anchor considering all its neighbors except itself
    min_dist = pairwise_dist + tf.to_float(diag_to_inf_mask)
    min_dist = tf.reduce_min(min_dist, axis=1, keepdims=True)
    # Create a matrix of pairwise distances with only mimimum distance is set to 0 row-wise
    # Uses broadcasting where the 1st argument  has shape (batch_size, batch_size) and the 2nd (batch_size, 1)
    rank1_matrix = pairwise_dist - min_dist
    # Edit the previous matrix to flag only elements that have the minimal distance to each anchor row-wise
    rank1_matrix = tf.to_float(tf.equal(rank1_matrix, 0.0))
    # Create a mask for all negative nearest neighbors
    rank1_matrix_neg = tf.multiply(rank1_matrix, mask_anchor_negative)
    # Flag for each anchor indicating whether or not there is a negative as nearest neighbor
    rank1_vector_neg = tf.reduce_max(rank1_matrix_neg, axis=1, keepdims=True)
    # Create a mask for all positive nearest neighbors
    rank1_matrix_pos = tf.multiply(rank1_matrix, mask_anchor_positive)
    # Flag for each anchor indicating wether or not there is a positive as nearest neighbor 
    rank1_vector_pos = tf.reduce_max(rank1_matrix_pos, axis=1, keepdims=True)
    # Flag for each anchor if the nearest neighbor(s) is/are only positive
    rank1_vector = tf.maximum(rank1_vector_pos - rank1_vector_neg, 0)
    # Average the rank-1 accuracy on the whole batch
    rank1_acc = tf.reduce_mean(rank1_vector)

    return rank1_acc


def batch_all_triplet_loss(labels, embeddings, params):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and consider the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        params: contains hyperparameters of the model (ex: `params.minimize_max`)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=params.squared)

    # Compute the rank-1 accuracy
    rank1_acc = _rank1_accuracy(pairwise_dist, labels)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    if params.triplet_loss == "original":
        triplet_loss = anchor_positive_dist - anchor_negative_dist + params.margin
    elif params.triplet_loss == "log_ratio":
        triplet_loss = - tf.log((anchor_negative_dist + 1e-16) / ((anchor_positive_dist + 1e-16) * params.radial_margin))
    else:
        raise ValueError("Triplet loss parameter unkown: {}".format(params.triplet_loss))

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    positive_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Define max and min batch triplet loss
    max_batch_triplet_loss = tf.reduce_max(triplet_loss)
    min_batch_triplet_loss = tf.reduce_min(triplet_loss)

    # Get final triplet loss over the positive valid triplets
    if not params.minimize_max:
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    else:
        triplet_loss = tf.reduce_max(triplet_loss)

    output = {'triplet_loss': triplet_loss,
              'fraction_positive_triplets' : fraction_positive_triplets,
              'max_batch_triplet_loss' : max_batch_triplet_loss,
              'min_batch_triplet_loss' : min_batch_triplet_loss,
              'rank1_accuracy': rank1_acc}

    return output


def batch_hard_triplet_loss(labels, embeddings, params):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        params: contains hyperparameters of the model (ex: `params.minimize_max`)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=params.squared)

    # Compute the rank-1 accuracy
    rank1_acc = _rank1_accuracy(pairwise_dist, labels)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label and be distinct)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size, 1)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if params.triplet_loss == "original":
        triplet_loss =tf.maximum(hardest_positive_dist - hardest_negative_dist + params.margin, 0)
    elif params.triplet_loss == "log_ratio":
        triplet_loss = - tf.log((hardest_negative_dist + 1e-16) / ((hardest_positive_dist + 1e-16) * params.radial_margin))
        triplet_loss = tf.maximum(triplet_loss, 0.0)
    else:
        raise ValueError("Triplet loss parameter unkown: {}".format(params.triplet_loss))

    # Count number of positive triplets (where triplet_loss > 0)
    positive_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.to_float(tf.shape(triplet_loss)[0])
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Define max and min batch triplet loss
    max_batch_triplet_loss = tf.reduce_max(triplet_loss)
    min_batch_triplet_loss = tf.reduce_min(triplet_loss)

    # Get final triplet loss over all hardest triplets
    if not params.minimize_max:
        triplet_loss = tf.reduce_mean(triplet_loss)
    else:
        triplet_loss = tf.reduce_max(triplet_loss)

    output = {'triplet_loss': triplet_loss,
              'fraction_positive_triplets': fraction_positive_triplets,
              'max_batch_triplet_loss' : max_batch_triplet_loss,
              'min_batch_triplet_loss' : min_batch_triplet_loss,
              'hardest_positive_dist' : tf.reduce_max(hardest_positive_dist),
              'hardest_negative_dist' : tf.reduce_min(hardest_negative_dist),
              'rank1_accuracy': rank1_acc}

    return output


def batch_hard_pos_triplet_loss(labels, embeddings, params):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and all negatives to form triplets.
    We consider the loss only over the positive triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        params: contains hyperparameters of the model (ex: `params.minimize_max`)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=params.squared)
    
    # Compute the rank-1 accuracy
    rank1_acc = _rank1_accuracy(pairwise_dist, labels)
    
    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label and be distinct)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get all negatives and their distances
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We put to 0 any element that is not (a, n), shape (batch_size, batch_size)
    anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)


    print("* anchor_negative_dist shape: {}".format(anchor_negative_dist.shape))
    print("* hardest_positive_dist shape: {}".format(anchor_positive_dist.shape))

    x = tf.constant([[0,0], [1,1]])
    x = tf.reduce_max(x, axis=1)
    print(x)


    # Compute a 2D tensor of size (batch_size, batch_size)
    # triplet_loss[i, k] will contain the triplet loss of anchor=i, negative=k with biggest d(a, p)
    # Uses broadcasting where the 1st argument has shape (batch_size, 1)
    # and the 2nd (batch_size, batch_size)
    # triplet loss will be 0 if i and k give a non-valid anchor-negative (same labels)
    if params.triplet_loss == "original":
        triplet_loss = tf.maximum((hardest_positive_dist + params.margin) * mask_anchor_negative - anchor_negative_dist, 0.0)
    elif params.triplet_loss == "log_ratio":
        triplet_loss = - tf.log((anchor_negative_dist + 1e-16) / ((hardest_positive_dist * mask_anchor_negative + 1e-16) * params.radial_margin))
        triplet_loss = tf.maximum(triplet_loss, 0.0)
    else:
        raise ValueError("Triplet loss parameter unkown: {}".format(params.triplet_loss))

    # Count number of positive triplets (where triplet_loss > 0)
    positive_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(mask_anchor_negative)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Define max and min batch triplet loss
    max_batch_triplet_loss = tf.reduce_max(triplet_loss)
    min_batch_triplet_loss = tf.reduce_min(triplet_loss)

    # Get final triplet loss over the positive valid triplets
    if not params.minimize_max:
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    else:
        triplet_loss = tf.reduce_max(triplet_loss)

    output = {'triplet_loss': triplet_loss,
              'fraction_positive_triplets': fraction_positive_triplets,
              'max_batch_triplet_loss' : max_batch_triplet_loss,
              'min_batch_triplet_loss' : min_batch_triplet_loss,
              'hardest_positive_dist' : tf.reduce_max(hardest_positive_dist),
              'rank1_accuracy': rank1_acc}

    return output
