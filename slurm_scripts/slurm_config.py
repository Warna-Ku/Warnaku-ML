import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from ray import tune
import os

# Define SlurmConfig class
class SlurmConfig:
    def __init__(self,
                 input_size, image_transform, 
                 image_transform_inference,  # image transform specific for inference
                 target_transform,
                 use_weighted_loss,  # True in order to use weighted cross entropy loss
                 optimizer,
                 local_dir,  # parameter local_dir of function tune.run
                 tunerun_cfg,  # dictionary to be passed to parameter config of function tune.run
                 is_hpo_cfg):  # specifies whether the configuration is for hpo or not
        self.config_dict_ = {
            'input_size': input_size,
            'image_transform': image_transform,
            'image_transform_inference': image_transform_inference,
            'target_transform': target_transform,
            'weighted_loss': use_weighted_loss,
            'optimizer': optimizer,
            'local_dir': local_dir,
            'tunerun_cfg': tunerun_cfg,
            'hpo_cfg': is_hpo_cfg,
        }

    def config_dict(self):
        return self.config_dict_

# Global configurations

# config for training of demo models
GLOBAL_INPUT_SIZE_TRAINING_DEMO = (256, 256)
GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(*GLOBAL_INPUT_SIZE_TRAINING_DEMO),
    layers.experimental.preprocessing.Rescaling(1./255)
])
GLOBAL_CFG_TRAINING_DEMO = SlurmConfig(
    GLOBAL_INPUT_SIZE_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    None,  # Placeholder for target_transform
    False,
    tf.keras.optimizers.Adam,
    "./demo_path",  # Placeholder for local_dir
    {
        "lr": 0.01,
        "lr_scheduler": "none",
        "batch_size": 32,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./demo_path") + '/'
    },
    False
).config_dict()

# FastSCNN-specific configurations

# config for hpo
FASTSCNN_INPUT_SIZE_HPO = (256, 256)
FASTSCNN_CFG_HPO = SlurmConfig(
    FASTSCNN_INPUT_SIZE_HPO,
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*FASTSCNN_INPUT_SIZE_HPO),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*FASTSCNN_INPUT_SIZE_HPO),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    None,  # Placeholder for target_transform
    True,
    tf.keras.optimizers.Adam,
    "./hpo_path",  # Placeholder for local_dir
    {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./hpo_path") + '/'
    },
    True
).config_dict()

# config for training with best hyperparameter values from hpo
FASTSCNN_INPUT_SIZE_TRAINING_BEST = (256, 256)
FASTSCNN_CFG_TRAINING_BEST = SlurmConfig(
    FASTSCNN_INPUT_SIZE_TRAINING_BEST,
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*FASTSCNN_INPUT_SIZE_TRAINING_BEST),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*FASTSCNN_INPUT_SIZE_TRAINING_BEST),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    None,  # Placeholder for target_transform
    True,
    tf.keras.optimizers.Adam,
    "./checkpoints_path",  # Placeholder for local_dir
    {
        "lr": 0.001,
        'lr_scheduler': "none",
        "batch_size": 16,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./checkpoints_path") + '/'
    },
    False
).config_dict()

# UNet-specific configurations

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
UNET_CFG_HPO = SlurmConfig(
    UNET_INPUT_SIZE_HPO,
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*UNET_INPUT_SIZE_HPO),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(*UNET_INPUT_SIZE_HPO),
        layers.experimental.preprocessing.Rescaling(1./255)
    ]),
    None,  # Placeholder for target_transform
    True,
    tf.keras.optimizers.Adam,
    "./hpo_path",  # Placeholder for local_dir
    {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune
    }
