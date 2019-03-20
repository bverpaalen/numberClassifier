import numpy as np


trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

inputTraining = np.loadtxt(trainDataIn, delimiter = ',')
outputTraining = np.loadtxt(trainDataOut, delimiter = ',')

inputTest = np.loadtxt(testDataIn, delimiter = ',')
outputTest = np.loadtxt(testDataOut, delimiter = ',')


class Perceptron():
    
    #Contructor for initializing the object:
    def __init__(self, inputData, outputData, weights):
        self.inputData = self.addBias(inputData)
        self.outputData = outputData
        self.weights = self.initialWeights(inputData)
        
    def addBias(self, inputData):
        a, b = np.shape(inputData)
        c = np.ones((a, 1))
        return np.hstack((c, inputData))
    
    def initialWeights(self, dim, classes):
        weights = np.random.rand(dim, classes)
        return np.array(weights)
    
    
    #The Perceptron Algorithm for computing the output of the perceptron
    #with inputData as the activation for each input node of the perceptron:
    def feedForward(self, inputData, outputData, weights):
        a, b = np.shape(inputData)
        weightedSum = np.zeros(257, 10)
        predictedOutput = np.zeros(257, 10)
        
        for i in range(0, len(a)):
            weightedSum[i,:] = np.dot(inputData[i,:], np.transpose(weights))
            predictedOutput[i,:] = self.sigmoid(weightedSum)
        
        return predictedOutput

    def sigmoid(x):
        return 1 / (1 + np.math.e ** -x)
    
    
    def updateWeight(weights):
        

#Training on 1707 images in the training set of the MNIST data:
Train = Perceptron()

iTrain = Train.inputData
oTrain = Train.outputData

#Initialize weights:
dimensions = 256 + 1 #the dimensions of a 16*16 sized image & 1 as a bias for the perceptron
classes = 10
weight_matrix = Train.weights

prediction = Train.feedForward(iTrain, oTrain, weight_matrix)
count = 1
misclassify = 0

while(count > 0 or iterations > 1000):
    count = 0
    for i in range(0, len(oTrain)):
        if oTrain == prediction:
            continue
        weight_matrix[i] = Train.updateWeight(weight_matrix[i])
        count += 1
        misclassify += 1
    
    iterations += 1
    print('Number of misclassified cases in 1000 iterations:' + misclassify)