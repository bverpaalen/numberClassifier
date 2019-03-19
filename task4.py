import numpy as np
import pandas as pd


trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

inputTraining = np.loadtxt(trainDataIn, dtype = 'str', delimiter = ',')
outputTraining = np.loadtxt(trainDataOut, dtype = 'str', delimiter = ',')

inputTest = np.loadtxt(testDataIn, dtype = 'str', delimiter = ',')
outputTest = np.loadtxt(testDataOut, dtype = 'str', delimiter = ',')

class Perceptron:
    
    #Contructor for initializing the object:
    def __init__(self, inputData, outputData, learning = 0.1, weights):
        self.inputData = addBias(inputData)
        self.outputData = outputData
        self.learning = learning
        self.weights = initialWeights(inputData)
        
    def addBias(self, inputData):
        a, b = np.shape(inputData)
        c = np.ones((a, 1))
        return np.hstack(c, inputData)
    
    def initialWeights(self, inputData):
        a, b = np.shape(inputData)
        
        for i in range(0, 10):
            weights[:, i] = np.ramdom.rand(b, 1)
        
        return np.asarray(weights)
    
    
    #The Perceptron Algorithm for computing the output of the perceptron
    #with inputData as the activation for each input node of the perceptron:
    def feedForward(self, inputData, outputData, weights):
        a, b = np.shape(inputData)
        
        for i in range(0, len(a)):
            weightedSum[i,:] = np.dot(weights, inputData[i,:])
            activations[i,:] = sigmoid(weightedSum)
            x, predictedOutput = np.shape(np.argmax(activations)) #Determines the image class in predictedOutput
        
        return predictedOutput, activations

    def sigmoid(x):
        return 1 / (1 + np.math.e ** -x)
    
    
    #Using backpropagation algorithm to update weights accordingly:
    def backPropagation(self, outputData, predictedOutput, activations, weights):
        
        for j in range(0, len(outputData))):
            
            for i in range(0, 10):
                
                if i == outputData[j]:
                    desiredActivations[j, i] = 1
                else:
                    desiredActivations[j, i] = 0
            
        for j in range(0, len(outputData)):
            
            cost[j, :] = np.square(activations[j, :] - desiredActivations[j, :])
            for i in range(0, 10):
                
                if cost[j, i] > 0:
                    weights = 
