import numpy as np
import math as m
import copy

epochs = 100
epsilon = 0.005
lr = 0.05

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 1, 1, 0]

class network():

    def __init__(self,inputSize,depth,width):
        trueWidth = max(depth,width)+1
        self.inputNodes = np.empty(inputSize+1)
        self.inputNodes[0] = 1
        self.hiddenNodes = np.empty(depth,width+1)
        self.hiddenNodes[:][0] = 1
        self.initialWeights(depth+1,trueWidth)

    def initialWeights(self,depth,width):
        np.random.seed(200)
        self.weights = np.random.rand(depth, width)

    def predict(self,input):
        for i in range(2):
            self.inputNodes[i+1] = input[i]
        self.calculateHiddenNodes()
        self.calculateOutputNode()

    def calculateHiddenNodes(self):

        for i in range(len(self.hiddenNodes)):
            layer = self.hiddenNodes[i]
            for j in range(len(layer)):
                layer[j] = 0
                if i == 0:
                    for k in range(len(self.inputNodes)):
                        layer[j] += self.weights[0][k] * self.inputNodes[k]
                    layer[j] = 1 / (1 + m.e ** -layer[j])
                else:
                    for k in range(len(self.hiddenNodes[j-1])):
                        layer[j] += self.weights[i][k]
                    layer[j] = 1 / (1 + m.e ** -layer[j])
            self.hiddenNodes[i] = layer

    def calculateOutputNode(self):
        outputNode = 0
        for i in range(len(self.hiddenNodes[-1])):
            outputNode += self.weights[-1][i]
        outputNode = outputNode/len(self.hiddenNodes[-1])

        return outputNode

    def mse(self,data):
        SE = 0
        for i in range(len(data)):
            prediction = data[i][0]
            target = data[i][1]

            SE += (prediction - target)**2
        MSE = SE/range(data)
        return MSE

    def grdmse(self,data):
        mse = self.mse(data)
        #GRDMSE function

    def changeWeights(self):
        #WEIGHT change function

net = network(2,2,2)

for epoch in range(epochs):
    for i in range(len(inputs)):
        label = targets[i]
        prediction = net.predict(inputs[i])



