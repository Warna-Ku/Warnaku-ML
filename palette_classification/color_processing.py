import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import utils.utils as utils
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings
from skimage import color

def color_distance(color1_RGB, color2_RGB):
    assert(color1_RGB.shape == (3, 1, 1) and color2_RGB.shape == (3, 1, 1))

    color1_RGB_np_HWD = utils.from_DHW_to_HWD(color1_RGB).numpy()
    color2_RGB_np_HWD = utils.from_DHW_to_HWD(color2_RGB).numpy()
    color1_CIELab = color.rgb2lab(color1_RGB_np_HWD)
    color2_CIELab = color.rgb2lab(color2_RGB_np_HWD)
    return np.linalg.norm(color1_CIELab - color2_CIELab)
    
def color_mask(img, color_triplet=[0, 0, 0]):
    assert(img.shape[0] == 3 and len(color_triplet) == 3)

    ch0, ch1, ch2 = color_triplet
    mask = (img[0] == ch0) & (img[1] == ch1) & (img[2] == ch2)
    return mask

def compute_segmentation_masks(img_segmented, labels):
    n_labels = len(labels)
    _, H, W = img_segmented.shape
    masks = tf.zeros((n_labels, H, W), dtype=tf.bool)

    for idx, label in enumerate(labels):
        label_color = labels[label]
        masks = tf.tensor_scatter_nd_update(masks, [[idx]], [color_mask(img_segmented, label_color)])

    return masks

def erode_segmentation_mask(segmentation_mask, kernel_size):
    assert(segmentation_mask.shape[0] == 1 and len(segmentation_mask.shape) == 3)

    _, H, W = segmentation_mask.shape
    kernel = cv2.getStructuringElement(shape=0, ksize=(kernel_size, kernel_size))

    extended_segmentation_mask = tf.tile(segmentation_mask, [3, 1, 1])
    img_binarized = tf.where(extended_segmentation_mask, 255, 0)
    img_binarized_eroded = cv2.erode(utils.from_DHW_to_HWD(img_binarized).numpy(), kernel=kernel)
    img_binarized_eroded = utils.from_HWD_to_DHW(tf.convert_to_tensor(img_binarized_eroded))
    img_binarized_eroded = tf.expand_dims(img_binarized_eroded, axis=0)
    img_binarized_eroded = tf.reduce_sum(img_binarized_eroded, axis=1)
    segmentation_mask_eroded = tf.where(img_binarized_eroded > 0, True, False)

    return segmentation_mask_eroded

def colorize_segmentation_masks(segmentation_masks, labels):
    assert(segmentation_masks.shape[0] == len(labels))

    n_labels = segmentation_masks.shape[0]
    color_tensor = tf.constant(list(labels.values()), dtype=tf.uint8)
    img_colorized = tf.reduce_sum(segmentation_masks[:, tf.newaxis, :, :] * color_tensor[:, :, tf.newaxis, tf.newaxis], axis=0)
    return tf.cast(img_colorized, tf.uint8)

def apply_masks(img, masks):
    assert(img.shape[1] == masks.shape[1] and img.shape[2] == masks.shape[2])

    img_masked = img * masks[:, tf.newaxis, :, :]
    return tf.cast(img_masked, tf.uint8)

def compute_candidate_dominants_and_reconstructions_(img_masked, n_candidates, return_recs=True):
    _, H, W = img_masked.shape
    kmeans = KMeans(n_clusters=n_candidates, n_init=10, random_state=99)
    mask_i = ~color_mask(img_masked)
    img_masked_i_flattened = utils.from_DHW_to_HWD(img_masked).reshape((H * W, -1)) / 255
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Number of distinct clusters*')
        kmeans.fit(img_masked_i_flattened)

    candidates = tf.round(tf.convert_to_tensor(kmeans.cluster_centers_) * 255)

    if return_recs is True:
        reconstructions = mask_i[:, tf.newaxis, :, :] * candidates[:, :, tf.newaxis, tf.newaxis]
        return candidates, reconstructions

    return candidates, None

def compute_user_embedding(img_masked, n_candidates, distance_fn, debug=False, eyes_idx=3):
    assert(img_masked.shape[:2] == (4, 3) and len(n_candidates) == 4)

    _, _, H, W = img_masked.shape
    dominants = []

    for i in range(4):
        max_brightness_i = cv2.cvtColor(utils.from_DHW_to_HWD(img_masked[i] / 255).numpy().astype(np.float32), cv2.COLOR_RGB2HSV)[:, :, 2].max()
        candidates, reconstructions = compute_candidate_dominants_and_reconstructions_(img_masked[i], n_candidates[i])

        min_reconstruction_error = -1 
        dominant = tf.zeros((3,), dtype=tf.uint8)

        for j, reconstruction_j in enumerate(reconstructions):
            if tf.reduce_sum(candidates[j]) < 20 or tf.reduce_sum(candidates[j]) > 700:
                continue
            
            average_brightness_j = cv2.cvtColor(utils.from_DHW_to_HWD(reconstruction_j / 255).numpy().astype(np.float32), cv2.COLOR_RGB2HSV)[:, :, 2].mean()
            reconstruction_error_j = distance_fn(img_masked[i], reconstruction_j).numpy()

            if i == eyes_idx:
                reconstruction_error_j *= (average_brightness_j / max_brightness_i)
            else:
                reconstruction_error_j /= (average_brightness_j / max_brightness_i)

            if debug is True:
                r, g, b = candidates[j]
                print(f'Candidate: ({r},{g},{b}), Weighted Reconstruction Error: {reconstruction_error_j}')
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(utils.from_DHW_to_HWD(reconstruction_j))
                plt.subplot(1, 2, 2)
                plt.imshow(utils.from_DHW_to_HWD(img_masked[i]))
                plt.show() 

            if min_reconstruction_error == -1 or reconstruction_error_j < min_reconstruction_error:
                min_reconstruction_error = reconstruction_error_j
                dominant = candidates[j]
            
        dominants.append(dominant.numpy().tolist())
    
    return tf.convert_to_tensor(dominants, dtype=tf.uint8).reshape((4, 3, 1, 1))

def compute_cloth_embedding(cloth_img_masked, max_length=10, ignored_colors=[]):
    assert(cloth_img_masked.shape[:2] == (1, 3))

    _, _, H, W = cloth_img_masked.shape
    embedding = []
    
    cloth_colors, _ = compute_candidate_dominants_and_reconstructions_(cloth_img_masked[0], max_length + 1, return_recs=False)

    for color in cloth_colors:
        for ignored_color in ignored_colors:
            color_triplet = color.numpy().tolist()
            
            if color_triplet != ignored_color:
                embedding.append(color_triplet)

    return tf.convert_to_tensor(embedding, dtype=tf.uint8).reshape(len(embedding), 3, 1, 1)

