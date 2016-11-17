Introduction to Neural Networks, University of Notre Dame (CSE 40868/60868)

# Assignment 2 (CNN)
In this assignment, we will experiment with the CIFAR-10 dataset and Convolutional Neural Networks. What you need to do:

## Core part
* Download data from here https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz. This data contains 50k images for training, and 10k for testing. 
* Write code to load the data properly. The training data is stored in the files ```data_batch_{1,2,3,4,5}``` and testing data is stored in ```test_batch```. Use ```cPickle``` to load each file into a Python dictionary object. The value with key ```data``` represents the image data, while that with ```labels``` represents the corresponding label. Each file contains 10k images, each has its pixel values presented as a row of 3072 elements, the first 1024 elements correspond to the red channel, then next 1024 elements correspond to the green channel, and last 1024 elements correspond to the blue channel. Load the image data into a numpy array and transpose it into shape ```[num_training, image_height, image_width, num_channels]```. We have 50k training samples in total, split it into 49k for training and 1k for validation. For the testing we have 10k images, but do not use them in this core part.
* Another step of preprocessing could be subtracting all images by the mean of all training images. 
```python
mean_image = numpy.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
``` 
* Using tensorflow, construct a shallow ConvNet of [conv layer + relu] + [max pool] + [fully connected + relu] + [softmax]
	- The conv layer should have 32 filters, filter size = 3
	- The max pooling layer of pool width = 2 and stride = 2
	- The fully connected layer has 512 hidden units
	- Use optimizer of your choice

## Your tasks
* Implement the above ConvNet
* Train this shallow ConvNet for 20 epochs; Report the loss & accuracy on validation set
* Report performance of the best model (according to performance on validation set) on the test set (should be >= 60%)
* Try different architectures (make it deeper, add dropout)
* Write a report (.pdf format) summarizing your steps and results. Submit it along with your source code.
* A README.md describing how to run your code. Please make it as easy as possible for me to run and see your reported results.
    * What version of Python/Tensorflow are you using?
    * What command to run to get the reported result for the core part?
    * What command to run to get the reported result for the bonus part?
    * ...


## Additional (the rest 5%)
* Visualize the learned filters. Can you provide any interpretation of the learned filters? (for instance: edge-detector-like, averaging, etc.)

