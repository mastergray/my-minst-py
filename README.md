# my-minst-py

Implements a class for working with a normalized MINST dataset

## What is MINST?

  From ChatGPT:

> The MNIST dataset, which stands for "Modified National Institute of Standards and Technology," is a widely used dataset in the field of machine learning and computer vision. It is a collection of handwritten digits, with each digit ranging from 0 to 9. The MNIST dataset is commonly used for tasks such as digit recognition, classification, and image analysis. It is a fundamental dataset often used to benchmark and develop machine learning algorithms, especially in the context of deep learning and neural networks.


## Overview 

- Each MINST sample is a 28x28 image of a number 0-9
- There are 60,000 training samples
- There are 10,000 testing samples
- MINST samples aren't stored as pixels of the image, but a single value for greyscale intensity
- Possible labels are 0 - 9, where each number represents the number being shown by the sample, i.e. a label of "1" means the sample is a picture of the number "1" 

For **my-minst-py**:

- Training and testing data are saved as [npy](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) using the MINST dataset provided by keras:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

- Samples have been normalized between [0,1] using:

```python
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
```

## Examples

```python 

from myminst import myminst 

trainingSet = myminst.initTrainingSet()    # Initialize training set
print(trainingSet.sample(0))               # Prints the first sample from the training set
print(trainingSet.label(0))                # Print the first label of the first label of the training set
print(trainingSet.value(0, [10,10]))       # Print the grey scale intestity at position of (10,10) for the first sample
trainingSet.plotSamples(10)                # Plots the first 10 samples of the loaded traing set 

```

## Notes 

- When loading MINST dataset from keras using:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train[0][0][0]     # First "pixel" of first image
x_train[0][27][27]   # Last "pixel" of first image
y_train[0]           # label for the first sample
```

- Traning samples had to be split into two files because of Github's 100MB file limit - but loading these files is handled by the static `initTraningSet` method 


## References

- [Keras example CNN model for MINST](https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py)
