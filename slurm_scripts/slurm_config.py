import tensorflow as tf
import tensorflow_transform as tft
from models import config
from ray import tune
import os


class SlurmConfig:
    def __init__(self,
                 input_size, image_transform,
                 image_transform_inference,  # image transform specific for inference
                 target_transform,
                 use_weighted_loss,  # True in order to use weighted cross entropy loss
                 optimizer,
                 local_dir,  # parameter local_dir of function tune.run
                 tunerun_cfg,  # dictionary to be passed to parameter config of function tune.run
                 is_hpo_cfg  # specifies whether the configuration is for hpo or not)
                 ):
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


# === UNet-specific configurations ===

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
tf.keras.Sequential([
    tf.keras.layers.Resizing(UNET_INPUT_SIZE_HPO),
    tf.keras.layers.Normalization(mean=config.NORMALIZE_MEAN, variance=config.NORMALIZE_STD)])
UNET_CFG_HPO = SlurmConfig(
    UNET_INPUT_SIZE_HPO,
    tf.keras.Sequential([
        tf.keras.layers.Resizing(UNET_INPUT_SIZE_HPO),
        tf.keras.layers.Normalization(mean=config.NORMALIZE_MEAN, variance=config.NORMALIZE_STD)]),
    tf.keras.Sequential([
        tf.keras.layers.Resizing(UNET_INPUT_SIZE_HPO),
        tf.keras.layers.Normalization(mean=config.NORMALIZE_MEAN, variance=config.NORMALIZE_STD)]),
    tf.keras.Sequential([tf.keras.layers.Resizing(UNET_INPUT_SIZE_HPO)]),
    True,
    tf.keras.optimizers.Adam,
    config.HPO_PATH,
    {
        "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
        "lr_scheduler": tune.grid_search(["none", "linear"]),
        "batch_size": tune.grid_search([16, 32]),
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.HPO_PATH) + '/'
    },
    True
).config_dict()

# config for training with best hyperparameter values from hpo
UNET_INPUT_SIZE_TRAINING_BEST = (256, 256)
UNET_CFG_TRAINING_BEST = SlurmConfig(
    UNET_INPUT_SIZE_TRAINING_BEST,
    tf.keras.Sequential([
        tf.keras.layers.ColorJitter(brightness=0.25, contrast=0.25),
        tf.keras.layers.Resizing(UNET_INPUT_SIZE_TRAINING_BEST),
        tft.custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        tf.keras.layers.Normalization(mean=config.NORMALIZE_MEAN, variance=config.NORMALIZE_STD)]),
    tf.keras.Sequential([
        tf.keras.layers.Resizing(UNET_INPUT_SIZE_TRAINING_BEST),
        tft.custom_transforms.BilateralFilter(sigma_color=50, sigma_space=100, diameter=7),
        tf.keras.layers.Normalization(mean=config.NORMALIZE_MEAN, variance=config.NORMALIZE_STD)]),
    tf.keras.Sequential([tf.keras.layers.Resizing(UNET_INPUT_SIZE_TRAINING_BEST)]),
    True,
    tf.keras.optimizers.Adam,
    config.CHECKPOINTS_PATH,
    {
        "lr": 1e-4,
        'lr_scheduler': "none",
        "batch_size": 16,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH) + '/'
    },
    False
).config_dict()

# slurm configurations
SLURM_CFG_HPO = {'unet': UNET_CFG_HPO}
SLURM_CFG_TRAINING_BEST = {'unet': UNET_CFG_TRAINING_BEST}

# dictionary containing all slurm configurations
configurations = {'hpo': SLURM_CFG_HPO, 'best': SLURM_CFG_TRAINING_BEST}
