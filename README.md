# Urban Sound Classification
Sw project for the course of Statistical Methods for Machine Learning, Master Degree in Computer Science at Unimi, a.y 20/21 

## Task
Use TensorFlow 2 to train neural networks for the classification of sound events based on audio files from the UrbanSound8K dataset: it contains 8732 sound excerpts (<=4 seconds) of urban sounds labeled with 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.  
The dataset is provided with 10 predefined folds. Train the model on folds: 1, 2, 3, 4, 6, and test it on folds: 5, 7, 8, 9, 10. Report the obtained average accuracy and standard deviation across the test folds. Experiment with different feature extraction methods, network architectures and training parameters documenting their influence of the final predictive performance.  

## Overview
For each audio file I've extracted the Mel frequency cepstral coefficients, the chroma vector, the zero-crossing rate, the RMSE and its statistics, to construct a dataset with which I fed FFNNs, and its spectogram, saved as 64x64 pixels RGB image, to fed CNNs.  
Feature selection and feature extraction techniques such as Boruta and PCA proved ineffective and worsened the performance of the models.  
FFNNs performs better than CNNs while MMNN can be a competitive and very interesting kind of neural network to increase performance on this task.

## MMNN
A multimodal neural network is a neural network composed by different submodels that is able to capture multimodal data, i.e. different type of input, trying to combine them to improve the results for a certain task.
Below you will find my proposed MMNN Architecture:
![mmnn-arch](img/mmnn-arch.png)
There is the best CNN I found that is fed with spectograms and the best FFNN that is trained with audio features, then a layer that concatenates the last hidden layer output of the two models and finally a last dense hidden layer with 128 neurons followed by the output layer.  

## Results
MMNN, FFNN, CNN's accuracies in function of the number of the epoch:  
![best_nn_acc](img/best_nn_acc.png)
MMNN accuracy along the different test folds:
![mmnn_acc](img/mmnn_acc.png)
