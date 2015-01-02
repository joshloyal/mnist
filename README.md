Digit Recognition
=================
Implementation of a neural network tested using Kaggle's Digit Recognition competition.

Problem Description
-------------------
The goal in this competition is to take an image of a handwritten single digit, and determine what the digit is. The data taken for this competition were taken from the MNIST dataset. The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic within the Machine Learning community that has been extensively studied. More detail about the dataset can be found [here](http://yann.lecun.com/exdb/mnist/index.html).

The Algorithm
-------------
This model implements a fully vectorized multi-layer sigmoid feed-forward neural-network with L2 regularization using Numpy. The network is trained using stochastic gradient descent along with momentum. The images are standardized and then the first 100 principle components are fed into the network. 
