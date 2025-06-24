# Transfer Learning with MobileNet

This project demonstrates how to build and train a powerful binary image classifier using transfer learning. It leverages the pre-trained MobileNetV2 model, fine-tuning it on a custom dataset. The process includes data loading, augmentation, feature extraction, and fine-tuning for optimal performance.

The example code is structured to classify images of alpacas, but it can be easily adapted for any binary classification task.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset Setup](#dataset-setup)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
  - [Phase 1: Feature Extraction](#phase-1-feature-extraction)
  - [Phase 2: Fine-Tuning](#phase-2-fine-tuning)
- [Results](#results)
- [Prerequisites](#prerequisites)

## Project Overview

The goal of this project is to build a high-accuracy image classifier with a relatively small custom dataset. Instead of training a model from scratch, which requires vast amounts of data and computational power, we use **transfer learning**. We start with the MobileNetV2 model, pre-trained on the large ImageNet dataset, and adapt it to our specific task.

The script performs the following steps:
1.  Loads a custom image dataset from a directory.
2.  Applies data augmentation to increase the diversity of the training set.
3.  Builds a new model by adding a custom classification head on top of a frozen MobileNetV2 base.
4.  Trains the new classification layers.
5.  Unfreezes some of the top layers of the base model and fine-tunes them with a lower learning rate to further improve accuracy.
6.  Visualizes the training progress.

## Key Features

- **Transfer Learning**: Utilizes the powerful **MobileNetV2** architecture.
- **Data Augmentation**: Employs `RandomFlip` and `RandomRotation` to prevent overfitting and improve model generalization.
- **Two-Phase Training**: A robust training strategy involving initial feature extraction followed by fine-tuning.
- **Efficient Data Handling**: Uses `tf.data.Dataset` for efficient input pipelines, including prefetching.
- **Built with TensorFlow 2.x and Keras**.

## Dataset Setup

The script expects the dataset to be organized in a specific directory structure. Create a main `dataset/` directory, and inside it, create one subdirectory for each class.

For this project, the structure should look like this:

- `dataset/`
  - `class_a/` (e.g., alpacas)
    - `image_1.jpg`
    - `image_2.jpg`
    - `...`
  - `class_b/` (e.g., not_alpacas)
    - `image_x.jpg`
    - `image_y.jpg`
    - `...`

The script will automatically infer the class names from the subdirectory names and split the data into training (80%) and validation (20%) sets.

- **Image Size**: `(160, 160)` pixels
- **Batch Size**: `32`

## Model Architecture

The model is constructed by stacking a custom classifier on top of the MobileNetV2 base.

1.  **Inputs**: The model takes batches of `(160, 160, 3)` images.
2.  **Data Augmentation**: A `Sequential` model applies random horizontal flips and rotations.
3.  **MobileNetV2 Base**: The convolutional base of MobileNetV2 (with weights pre-trained on ImageNet) acts as a feature extractor. Its layers are initially frozen.
4.  **Classifier Head**:
    - A `GlobalAveragePooling2D` layer reduces the feature maps to a single vector per image.
    - A `Dropout` layer (rate=0.2) is applied to prevent overfitting.
    - A final `Dense` layer with a single output neuron produces the raw prediction (logit) for binary classification.

The model summary is as follows:

| Layer (type)               | Output Shape            | Param #   |
| -------------------------- | ----------------------- | --------- |
| `input`                    | (None, 160, 160, 3)     | 0         |
| `data_augmentation`        | (None, 160, 160, 3)     | 0         |
| `mobilenetv2_1.00_160`     | (None, 5, 5, 1280)      | 2,257,984 |
| `global_average_pooling2d` | (None, 1280)            | 0         |
| `dropout`                  | (None, 1280)            | 0         |
| `dense`                    | (None, 1)               | 1,281     |

**Total params**: 2,259,265
**Trainable params**: 1,281 (initially)

## Training Process

### Phase 1: Feature Extraction

In the first phase, we freeze all the layers of the MobileNetV2 base model. This ensures that their pre-trained weights are not modified. We only train the newly added classifier head (`GlobalAveragePooling2D`, `Dropout`, `Dense`). This allows the model to learn how to map the extracted features to the new classes.

- **Epochs**: 5
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss**: Binary Cross-Entropy (from logits)

### Phase 2: Fine-Tuning

After the classifier head has stabilized, we unfreeze the top layers of the MobileNetV2 base model (from layer 126 onwards). We then continue training the entire model with a much lower learning rate. This allows the model to "fine-tune" the high-level features from the base model for our specific dataset, leading to a significant accuracy boost.

- **Epochs**: 5 more (total 10)
- **Optimizer**: Adam
- **Learning Rate**: 0.0001 (10% of the initial rate)
- **Loss**: Binary Cross-Entropy (from logits)

## Results

The script generates plots to visualize the training and validation accuracy and loss over all epochs. The plots clearly show the two phases of training, with a distinct marker indicating the start of the fine-tuning phase.


*Example of the final output plot showing accuracy and loss curves.*

We typically expect to see a jump in validation accuracy after the fine-tuning phase begins.

## Prerequisites

You need Python 3 and the following libraries installed:
- `tensorflow`
- `matplotlib`
- `numpy`
