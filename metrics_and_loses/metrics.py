# --- Needed to import modules from other packages
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# ---

import cv2
import tensorflow as tf
import utils.utils as utils


def rmse(img1, img2):
    """
    .. description::
    Converts two images (tf.Tensor instances) of shape (D, H, W) in CIELab and then computes the RMSE between them.
    """
    assert(img1.shape == img2.shape)

    _, H, W = img1.shape
    img1_np_HWD = utils.from_DHW_to_HWD(img1).numpy()
    img2_np_HWD = utils.from_DHW_to_HWD(img2).numpy()
    img1_CIELab = cv2.cvtColor(img1_np_HWD, cv2.COLOR_RGB2Lab)
    img2_CIELab = cv2.cvtColor(img2_np_HWD, cv2.COLOR_RGB2Lab)

    return tf.sqrt(tf.reduce_sum(tf.square(img1_CIELab - img2_CIELab)) / (H * W))


def batch_mIoU(predictions, targets, weights=None):
    """
    .. description::
    Returns a tensorflow tensor containing the average mIoU along a batch of images, or
    the weighted average if a tensorflow tensor of weights is provided.

    .. inputs::
    predictions:    tensorflow tensor of shape (batch_size, n_labels, H, W).
    targets:        tensorflow tensor of shape (batch_size, n_labels, H, W).
    weights:        tensorflow tensor of shape (n_labels,).
    """
    assert(weights is None or (isinstance(weights, tf.Tensor) and weights.shape == (targets.shape[1],)))

    iou = batch_IoU(predictions, targets)

    if weights is None:
        return tf.reduce_mean(iou)

    return utils.tensor_weighted_average(iou, weights)


def batch_IoU(predictions, targets, _=None):
    """
    .. description::
    Returns a tensorflow tensor of shape (n_labels,) containing the IoU for each label along a batch of images.

    .. inputs::
    predictions:    tensorflow tensor of shape (batch_size, n_labels, H, W).
    targets:        tensorflow tensor of shape (batch_size, n_labels, H, W).
    """
    intersection_cardinality = tf.reduce_sum(tf.cast(tf.logical_and(predictions, targets), tf.float32), axis=(2, 3))
    union_cardinality = tf.reduce_sum(tf.cast(tf.logical_or(predictions, targets), tf.float32), axis=(2, 3))
    IoU = intersection_cardinality / union_cardinality

    # if there aren't pixels of a certain class in an image, and the model correctly predicts so, 
    # than the IoU should be 1 for that class
    IoU = tf.where(union_cardinality == 0, tf.ones_like(IoU), IoU)

    return tf.reduce_mean(IoU, axis=0)

