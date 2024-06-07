import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import ray
from ray import tune

class SlurmConfig:
    def __init__(self,
                 input_size, image_transform, 
                 image_transform_inference, 
                 target_transform,
                 use_weighted_loss,  
                 optimizer,
                 local_dir,  
                 tunerun_cfg,  
                 is_hpo_cfg  
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

# === Global configurations ===

# config for training of demo models
GLOBAL_INPUT_SIZE_TRAINING_DEMO = (256, 256)
GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO = ImageDataGenerator(rescale=1./255)
GLOBAL_CFG_TRAINING_DEMO = SlurmConfig(
    GLOBAL_INPUT_SIZE_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    GLOBAL_IMAGE_TRANSFORM_TRAINING_DEMO,
    ImageDataGenerator(rescale=1./255),
    False,
    tf.keras.optimizers.Adam,
    config.DEMO_PATH,
    {
        "lr": 0.01,
        "lr_scheduler": "none",
        "batch_size": 32,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.DEMO_PATH) + '/'
    },
    False
).config_dict()

# === FastSCNN-specific configurations ===

# config for hpo
FASTSCNN_INPUT_SIZE_HPO = (256, 256)
FASTSCNN_CFG_HPO = SlurmConfig(
    FASTSCNN_INPUT_SIZE_HPO,
    ImageDataGenerator(rescale=1./255),
    ImageDataGenerator(rescale=1./255),
    ImageDataGenerator(rescale=1./255),
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
FASTSCNN_INPUT_SIZE_TRAINING_BEST = (256, 256)
FASTSCNN_CFG_TRAINING_BEST = SlurmConfig(
    FASTSCNN_INPUT_SIZE_TRAINING_BEST,
    ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: custom_transforms.bilateral_filter(x, sigma_color=50, sigma_space=100, diameter=7)
    ),
    ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: custom_transforms.bilateral_filter(x, sigma_color=50, sigma_space=100, diameter=7)
    ),
    ImageDataGenerator(rescale=1./255),
    True,
    tf.keras.optimizers.Adam,
    config.CHECKPOINTS_PATH,
    {
        "lr": 0.001,
        'lr_scheduler': "none",
        "batch_size": 16,
        "from_checkpoint": False,
        "checkpoint_dir": os.path.abspath("./" + config.CHECKPOINTS_PATH) + '/'
    },
    False
).config_dict()

# === UNet-specific configurations ===

# config for hpo
UNET_INPUT_SIZE_HPO = (256, 256)
UNET_CFG_HPO = SlurmConfig(
    UNET_INPUT_SIZE_HPO,
    ImageDataGenerator(rescale=1./255),
    ImageDataGenerator(rescale=1./255),
    ImageDataGenerator(rescale=1./255),
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
    ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: custom_transforms.color_jitter(x, brightness=0.25, contrast=0.25)
    ),
    ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=lambda x: custom_transforms.bilateral_filter(x, sigma_color=50, sigma_space=100, diameter=7)
    ),
    ImageDataGenerator(rescale=1./255),
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
MODEL_NAMES_LIST = list(model_names.MODEL_NAMES.keys())
SLURM_CFG_HPO = {'fastscnn': FASTSCNN_CFG_HPO, 'unet': UNET_CFG_HPO}
SLURM_CFG_TRAINING_BEST = {'fastscnn': FASTSCNN_CFG_TRAINING_BEST, 'unet': UNET_CFG_TRAINING_BEST}
SLURM_CFG_TRAINING_DEMO = {model_name: GLOBAL_CFG_TRAINING_DEMO for model_name in MODEL_NAMES_LIST}

# dictionary containing all slurm configurations
configurations = {'demo': SLURM_CFG_TRAINING_DEMO, 'hpo': SLURM_CFG_HPO, 'best': SLURM_CFG_TRAINING_BEST}
