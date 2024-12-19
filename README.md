[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cZnpr7Ns)
# E4040 2024 Fall Project
## Deep Cluster
This code implements the unsupervised training of convolutional neural networks, as described in the report [E4040.2024Fall.SZQA.report.zs2699.yq2411.jz3849].

Moreover, we provide the evaluation protocol codes we used in the paper:
* Pascal VOC classification
* Linear classification on activations

## Requirements
To run the code in this project, ensure you have the following dependencies installed:

- **Python Installation**  
  Python version 3.7 or later is recommended for compatibility with the libraries used.

- **Required Libraries**  
  - [NumPy](https://numpy.org/) for numerical computations.
  - [TensorFlow/Keras](https://www.tensorflow.org/) version 2.0 or later for deep learning implementations.
  - [Matplotlib](https://matplotlib.org/) for inline plotting and visualizations.
  - [Scikit-learn](https://scikit-learn.org/) for clustering (KMeans) and PCA-based dimensionality reduction.

- **GPU Support (Optional)**  
  For training acceleration, ensure you have:
  - CUDA 11.2 or later installed on your system.
  - TensorFlow GPU-compatible version installed.

- **Datasets**  
  - CIFAR-10 dataset, which can be loaded automatically via TensorFlow/Keras datasets.
 
## Pre-trained Models

We provide pre-trained models with MobileNet and custom CNN architectures, available for download. These models are implemented in TensorFlow/Keras format and are compatible with grayscale and RGB input images.

The path of **MobileNet** model:
```
model=/model/mobilenet_cluster_model.h5
```
The path of **Custom CNN** model:
```
model=/model/custom_cluster_model.h5
```
The pre-trained models expect input data with the following preprocessing steps applied:
1. **Grayscale Conversion**  
2. **Normalization**  
3. **Augmentation**

## Running the unsupervised training
Unsupervised training can be launched by running cells in
```
program.ipynb
```
The overvieww of workflow is
```
Load Data -> Feature Extraction -> Clustering -> Pseudo-Supervised Classification -> Save/load Model
```
## Description of key functions of each file
```
1. Activation.py 
```
Identify and visualize the top N images that activate a specific filter in a convolutional layer of a pre-trained deep learning model, using CIFAR-10 as input data.
```
2. Firstlayer_activation.py 
```
Visualize the responses of all filters in a specified convolutional layer of a pre-trained model using real CIFAR-10 images.
```
3. Firstlayer_directweights.py 
```
Visualizes the filters (weights) of the first convolutional layer in a pre-trained model, displaying them as grayscale images.
```
4. Gradient_ascend.py 
```
Generate, optimize, and visualize activation patterns of a specific filter in a convolutional layer of a pre-trained model using gradient ascent with customizable input.
```
5. Linear_classification.py 
```
Evaluate the representation quality of different layers in a pre-trained MobileNet model by extracting activations, reducing dimensionality with PCA, and training a logistic regression classifier to measure classification accuracy.
```
6. data_processing.py 
```
Perform data preprocessing and augmentation for the CIFAR-10 dataset and visualizes both raw and preprocessed images with their corresponding class labels.
```
7. evaluate.py 
```
Evaluate a trained model's accuracy using true labels and visualizes its predictions compared to the ground truth.
# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```

```
