# Simple Neural Networks with Keras
---
# Installation

For this project, you only need CPU performance, so you do not need to worry about installing any of the GPU dependencies such as [cuda](http://www.nvidia.com/object/cuda_home_new.html) or [cuDNN](https://developer.nvidia.com/cudnn), but feel free to try anyway.

We will experiment some simple MLP for multi-class classification using Keras. "Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano.", i.e., it enables us to construct (some) neural networks with very few lines of code, and under the hood, it uses Tensorflow or Theano as backend.

To install the necessary packages for this project, use pip. In next assignments, we will use Tensorflow, so I recommend you to use Tensorflow as backend. It's recommended to use version 0.9 since our code is tested in 0.9, but feel free to use other versions (.10, .11). Please follow this [guide](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup) to install Tensorflow.

Then, make sure your installation works in python by importing Keras
```sh
$ python
$ >>> import numpy as np
$ >>> import tensorflow
...probably some gibberish about GPUs and CPUs...
$ >>> import keras
using Tensorflow backend.
```

If you don't get any error when importing, everything should be good.

## A quick note on numpy: 
The numpy array object is incredibly important to neural network computing and essentially any other scientific computing in python. It would behoove you to become familiar with it's basic functionality. Usually a quick search can lead you to it's great documentation on any function or object. asarray, traspose, zeros, ones, argmax, save, load and  concatenate are all good ones to have memorized, as well as the shape data member. Personally, I highly recommend you to look at this tutorial http://cs231n.github.io/python-numpy-tutorial/

# A wordy tutorial with the Iris dataset

In this tutorial, we will build and train a simple MLP on one of the most well-known datasets in pattern recognition, Fisher's Iris dataset from 1936(!). This datset includes sepal width and length and petal width and length for 50 iris flowers, classified into three species: *Iris setosa*, *Iris versicolour*, and *Iris virginica*. Your task is to build and train a simple neural network in Keras that classifies an unknown flower's sepal and petal information into a species of iris. The dataset includes 150 flowers, make sure to split the data intro 80% training 20% testing or else you will have no way of assessing your network. This data is taken from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Iris) 

This is the first tutorial, so I will go step by step. For some (or all) of you, this could be too wordy. If so, I'm sorry about that. Feel free to skip anything you know already.

#### Step 1 | Simple data processing 
Take a look at the data ```iris.dat```. It consists of 150 obversed samples, each has 4 features, the sepal/petal width/length, and the last value is the name of the class the sample should belong to. There are 3 classes in total, 'Iris-versicolor', 'Iris-virginia' and 'Iris-setosa'. It's safe to say that our input data is a 150x4 matrix. Since we're doing multiclass classification, let's encoder our output data with one-hot encoding. Say, if we index the classes as following:  
Iris-setosa = 0  
Iris-versicolor = 1  
Iris-virginica = 2    
then, for each sample, the output should be a vector of all zeros but the corresponding class. For example, if a sample belongs to 'Iris-setosa', its output is [1 0 0], if 'Iris-versicolor', then output is [0 1 0], and so on.

If your data is:  
```
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,2.8,5.6,2.2,Iris-virginica
6.4,3.2,4.5,1.5,Iris-versicolor
5.8,4.0,1.2,0.2,Iris-setosa
```  
then:
```python
x = [[5.1, 3.5, 1.4, 0.2],
	 [7.0, 3.2, 4.7, 1.4],
	 [6.4, 2.8, 5.6, 2.2],
	 [6.4, 3.2, 4.5, 1.5],
	 [5.8, 4.0, 1.2, 0.2]]
y = [[1, 0, 0],
	 [0, 1, 0],
	 [0, 0, 1],
	 [0, 1, 0],
	 [1, 0, 0]]
```

If you're not familiar with Python or programming in general, here are some suggestions to do the data processing:
* Declare a zero numpy ndarray of shape [150, 4] for x, type should be float
* Declare a zero numpy ndarray of shape [150, 3] for y, type should be int
* Read ```iris.dat``` line by line. For each line, remember to strip the newline character, and ignore the empty line (line with only newline character). If the line is not empty, use ```split(',')``` to split the line into a list of 5 values. The first 4 shall go to the corresponding row in ```x```. Given the 5th value, set the corresponding row in ```y``` with either ```[1, 0, 0]```, ```[0, 1, 0]``` or ```[0, 0, 1]```. Note that if you do simple file reading, everything is a string. We need ```x```'s elements to be float. You can use ```map(float, l)``` to convert all elements of a list ```l``` to float.
* We need data for training, but also data for evaluating. Let's split this data with a ratio of 80(train)/20(test).
* Remember to shuffle the training data before splitting/training. 😊

#### Step 2 | Building a simple network in Keras
The first part of this assignment is a gentle introduction to using a neural network package. Like programming languages, the more familiar you are with neural networks and the more packages that you've used, the easier it is to learn how to use new ones. 

There are two model options in Keras, we will be using the Sequential model (the other is the functional model API, which is a bit more in depth). First, import ```keras```, ```numpy```, and anything else you might want to use, then instantiate a Sequential object.
```python
import keras as K
from keras.layers import Dense
from keras.models import Sequential

network = Sequential()
```
As the name implies, the Sequential model builds up layer by layer into the model that you want. Since this problem is very simple, we're only going to use a single hidden layer here.
```python
# Telling the network that each input sample is of shape [4] and that we want to use
# sigmoidal activation
network.add(Dense(16,input_dim=4, activation='sigmoid'))
# We don't need to specify the input dim here because Keras is smart enough to realize the input is
# the same dimension as the previous output. And of course, we want our output is of shape [3]
network.add(Dense(3, activation='softmax'))
```
For each sample, the softmax layer returns a length-3 vector whose each value represents the probability of this sample belongs to the corresponding class. For example, if the softmax output is [0.2, 0.2, 0.6], then our network thinks that this sample is a 'Iris-virginica'. 

Given this insight of the softmax output and the reference output y, we can use crossentropy as loss, and our goal is to minimize this loss. With Keras, it takes only 3 lines to do this.

Next, we're going to compile the network and train it. Compiling can be done in one line, but there are several possible parameters. Again, we're keeping it simple in this example, so all we'll have to do is this:
```python
opt = 'SGD' # We'll use SDG as our optimizer
obj = 'categorical_crossentropy' # And we'll use categorical cross-entropy as the objective we're trying to minimize
network.compile(optimizer=opt, loss=obj, metrics=['accuracy']) # Include accuracy for when we want to test our net
```
To train we use the fit method:
```python
NEPOCHS=100
# The fit method returns a history object
history = network.fit(train_x,train_y, #input, target
	nb_epoch=NEPOCHS, #number of epochs, or number of times we want to train with the entire dataset
	batch_size=1, #batch size, or number of samples trained at one time
	verbose=1) #verbosity of 1 gives us medium output
```
That's it! The network is trained. Usually the network takes more than a couple seconds to train (hours or days rather), but here we can see the results instantly. To test our network, we're going to use the evaluate method:
```python
loss, acc = network.evaluate(test_x, test_y) # Returns the loss and accuracy as a tuple
```

The general outline of the code can be:
```python
# Import data
x, y = importData()

# Split data
train_x = ...
train_y = ...
eval_x = ...
eval_y = ...

# Define the network
...

# Define the loss and compile
...

# Training & evaluating
train = ...
loss, acc = ...

# Print out result
print('Final accuracy after {} iterations: {}'.format(num_epochs, acc))
```
Depending on the number of epochs you chose, this should be up past 90. I get 70% - 100% accuracy when I train with 10 epochs. If I bump it to 100, I get 99%-100%.

### Last note
There are some practices that I omit. 
* Please note that neural networks are notorious about its data-hungry characteristic. Sometimes, you may find that with small dataset, the performance is really really bad. In our case, however, I find that using two layers (experiment with each layer's shape yourself) can result in high and stable accuracy.
* We use SGD for optimizer here, but you can try [others](https://keras.io/optimizers/). Adaptive learning rate optimizer such as Adadelta is very promising.
* If you use SGD, it's best to fine tune the learning rate. Some train for a certain number of epochs with fixed learning rate, then halve the learning rate after every epoch after that. Some do evaluation every now and then during the training, then look at the past evaluations and determine if the performance is not getting better, and reduce the learning rate. This is very tiresome but works really well in practice. 
* In practice, we should ensure the training data is well shuffled. It's best to shuffle before each epoch. If your data is small, you can load them all into memory, i.e. store it in a numpy.ndarray for example, and shuffle the array. 
* We normally have training data and test data. Training data is the only thing we know about the data, we train on it, we optimize the network to learn the best from it. But our goal is not to find the best model that fits this traning data. Our goal is to find a model that performs the best when it goes to the wild world out there, i.e. find the model that does the best on the test data. To find the best model that works well on test, but not touch the test data, we often split the data into 3 parts, namely train/dev/test. While optimizing the network on training data, every now and then we evaluate the model with dev data. Say you train for 30 epochs, and do evaluation after each epoch, so you have 30 models to pick from. What's the good spliting ratio? You may start with 80/10/10 proportion for train/dev/test.
* We often train neural networks with mini-batches. That is we divide the training data into many batches and for each training iteration, we feed a batch to the networks, calculate the loss, backpropagate the gradients and update the parameters. The reason for this roots in the insight of stochastic gradient descent algorithm, but we won't go deeper into this. The batch size is typically small, ranging from 1 to a few hundreds. Increasing the batch size can make training faster, but for many settings, small batch size results in better performance. Try varying the batch size and see it yourself. 

# Assignment 

## Assignment 1 | Iris
Finish above network for the Iris dataset. You can choose any number of layers, hidden layer sizes, training algorithm... as you want, as long as you get > 90% accuracy. 

## Assignment 2 | Wine
Similar to the Iris assignment but with the [Wine dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data). 
* Download the data. It has 179 samples, each is a row, the first value is the class and the rest 13 are features. 
* Build & train a MLP just like with Iris, try any number of layers, hidden layer sizes, algorithm...    

## Hand-in
A .zip or .tgz contains:  
* code
* A report (.pdf format) briefly describes what you do, results and any conclusion you may have.

Follow the course webpage for instruction of where to submit. 
