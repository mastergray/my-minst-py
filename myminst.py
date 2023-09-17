import numpy as np
import matplotlib.pyplot as plt

# Implements class for working with a normalized MINST dataset
class myminst:

    # CONSTRCTOR :: NUMPY, NUMPY -> self
    def __init__(self, samples, labels):
        self.samples = samples 
        self.labels = labels

    # :: NUMBER -> NUMPY
    # Return sample at given index:
    def sample(self,index):
        return self.samples[index]
    
    # :: NUMBER -> NUMBER
    # Returns label of sample at given index:
    def label(self, index):
        return self.labels[index]
    
    # :: NUMBER, [NUMBER, NUMBER] -> NUMBER
    # Returns value at x,y position of sample at given index:
    def value(self, index, position):
        x,y = position
        return self.samples[index][x][y]
    
    # :: NUMBER -> VOID
    # Plots the given number of sample images:
    def plotSamples(self, numberOfSamples):
        for i in range(numberOfSamples):
            plt.subplot(1, numberOfSamples, i + 1)
            plt.imshow(self.samples[i], cmap='gray')
            plt.title(f"Label: {self.labels[i]}")
            plt.axis('off')
        plt.show()
    
    # Static Factory Method :: [NUMPY], [NUMPY] -> minst
    # Loads sample and labels from npy files for given filepaths:
    @staticmethod
    def init(samples, labels):
        return myminst(samples, labels)
        
    # :: VOID -> minst
    # Loads training dataset
    @staticmethod 
    def initTrainingSet():
        samples_1 = np.load("./datasets/train_samples_1.npy")
        samples_2 = np.load("./datasets/train_samples_2.npy")
        samples = np.concatenate((samples_1, samples_2), axis=0)
        labels = np.load("./datasets/train_labels.npy")
        return myminst.init(samples, labels)

    # :: VOID -> minst
    # Loads training dataset
    @staticmethod 
    def initTestingSet():
        samples = np.load("./datasets/test_samples.npy")
        labels = np.load("./datasets/test_labels.npy")
        return myminst.init(samples, labels)


    # :: NUMPY -> NUMPY 
    # Flattens sample into 1D vector for comparison:
    @staticmethod
    def flatten(sample):
        return sample.reshape(-1)

    # :: NUMPY -> NUMPY 
    # Flattens array of samples into 1D vectors:
    @staticmethod
    def flattenAll(samples):
        return samples.reshape(samples.shape[0], -1)



