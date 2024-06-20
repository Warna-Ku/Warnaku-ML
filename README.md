# Warnaku-ML

## Notebooks
The `notebooks` folder contains two Jupyter notebooks:

1. **Model Training.ipynb:** This notebook is used for training the deep learning model for face segmentation and color analysis. It includes code for data preprocessing, model architecture, training, and evaluation.
2. **Palette Classification.ipynb:** This notebook contains the code for classifying the user's skin, hair, and eye colors into specific color palettes or seasonal color categories. It utilizes the trained model from `Model Training.ipynb` to perform color analysis on user images.

## Models
The `models` folder contains various formats of the trained model after the training process. These models are used by the WarnaKu application for color analysis and recommendations. The trained model is saved in the following formats:

- **H5:** This is the default file format for saving Keras models. It includes the model architecture, weights, and training configuration.
- **Keras:** This format saves the model in a Python-friendly format, allowing easy loading and integration into other Python applications.
- **SavedModel:** This is a portable, serialized format for TensorFlow models, which enables deployment on various platforms and environments.
- **Weight Files:** These are individual files containing the trained weights of the model, which can be loaded and used with the model architecture.

By providing the trained model in multiple formats, we ensure compatibility with various deployment scenarios and facilitate easier integration into the WarnaKu mobile application.

## Usage
To get the WarnaKu machine learning model, follow these steps:

1. Clone the repository: ```git clone https://github.com/Warna-Ku/Warnaku-ML.git```
2. If you want to train your own model, open the `Model Training.ipynb` notebook and follow the instructions to train the deep learning model for face segmentation. Alternatively, go to the `models` folder and download one of the saved models.
3. Open the `Palette Classification.ipynb` notebook and follow the instructions to classify user images into specific color palettes or seasonal color categories using the trained model.
4. Integrate the color analysis functionality into the WarnaKu mobile application or web interface for end-users.