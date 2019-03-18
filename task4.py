import numpy as np
import pandas as pd


trainDataPath = "D:/Career/Leiden/Masters/2nd Semester/Neural Networks/Assignments/Assignment 1/numberClassifier/data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "D:/Career/Leiden/Masters/2nd Semester/Neural Networks/Assignments/Assignment 1/numberClassifier/data/test_"
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
        
        return weights
    
    
    #The Perceptron Algorithm for computing the output of the perceptron
    #with inputData as the activation for each input node of the perceptron:
    def feedForward(self, inputData, outputData, weights):
        a, b = np.shape(inputData)
        
        for i in range(0, len(a)):
            weightedSum = np.dot(weights, inputData[i,:])
            x, predictedOutput = np.shape(np.argmax(weightedSum)) #Determines the image class in predictedOutput
            bestActivation = np.argmax(weightedSum)
        
        return predictedOutput, bestActivation

    
    #Evaluating the class of the image and update weights accordingly:
    def backPropagation(self, outputData, predictedOutput):
       
        for i in range(0, len(outputData)):
            
            if outputData[i] == predictedOutput[i]:
                prediction = 1
                else:
                    prediction = 0
            
            