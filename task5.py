import numpy as np
import math as m


epsilon = 0.001
epochs = 1000

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 1, 1, 0]
bias = [1, 1]

wMatrix1 = np.random.rand(3, 2)
wMatrix2 = np.random.rand(3, 1)
wMatrix = np.concatenate((wMatrix1, wMatrix2), axis = 1)


class xor_net():
    
    def __init__(self, x1, x2, weights):
        x1 = self.x1
        x2 = self.x2
        weightMatrix = self.weights
        
    def computeActivation(self, weight, inputData):
        act = np.argmax(np.dot(inputData, np.transpose(weight)))
        return self.sigmoid(act)
    
    def sigmoid(x):
        return 1 / (1 + m.e ** -x)
    
    
    def mse(self, weights):
        return np.mean((self.predictActivation(iData, weights) - oData)**2 for iData, oData in inputs)

    def predictActivations(self, iData, w):
        return self.sigmoid(w[7]*self.sigmoid(w[0]+w[1]*iData[0]+w[3]*iData[1]) + w[8]*self.sigmoid(w[5]+w[2]*iData[0]+w[4]*iData[1]) + w[6])