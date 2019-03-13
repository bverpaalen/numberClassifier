import numpy as np

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

class perceptron(object):
    
    def __init__(self, inputData, outputData, learning = 0.1):
        self.inputData = self.add_bias(inputData)
        self.outputData = outputData
        self.learning = learning
        
        self.weights = self.addWeights(inputTraining, outputTraining, learning)
        self.accuracy = self.accuracy(inputTest, outputTest, self.weights)
        
    def addBias(inputData):
        a, b = np.shape(inputData)
        c = np.ones((a, 1))
        return np.hstack(c, inputData)
        
    def initialWeights(self, inputData):
        a, b = np.shape(inputData)
        weights = np.ramdom.rand(b, 1)
        return weights
    
    def predictInitial(self, node, weights):
        activation = np.dot(node, weights)
        a = activation>0
        return a*1
    
    def addWeights(self, inputData, outputData, learning):
        c,d = np.shape(inputData)
        w = self.initialWeights(inputData)
        weights = []
        for i in range(0,  len(np.unique(outputData))):
            if (outputData == i):
                z = 1
            else:
                z = 0
            
            a = self.training(inputData, z, w, learning)
            weights.append(a[:,0])
        return np.asarray(weights)
    
    def updateWeights(self, inputData, weights):
        a = np.dot(inputData,np.transpose(weights))
        b = len(np.shape(inputData))
        if b == 1:
            return np.argmax(a)
        return np.argmax(a, axis=1)
    
    def accuracy(self, inputTest, outputTest, weights):
        x, y = np.shape(outputTest)
        predicted = self.updateWeights(inputTest, weights)
        correct = predicted == outputTest[:,0]
        accuracy = np.sum(correct)/float(x)
        return accuracy
    
    def reclassify(self, weights, node, output, learning = 0.1):
        if (np.dot(node, weights) > 0):
            predicted = 1
        else:
            predicted = 0
        temp = np.zeros(np.shape(weights))
        temp[:,0] = learning * np.square(output - predicted) * node
        return temp + weights
    
    def sigmoid(x):
        return 1 / (1 + np.math.e ** -x)
    
    def training(self, inputData, output, weights, learning):
        
        for i in range(0, len(inputData)):
            weights = self.reclassify(weights, inputData[i], output[i], learning)
        
        return weights