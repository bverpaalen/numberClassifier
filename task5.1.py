import random

import numpy as np
import math as m
import copy

epochs = 20000
epsilon = 0.005
lr = 0.5

possibilities = [[0, 0], [0, 1], [1, 0], [1, 1]]
inputs = []

for i in range(100):
    j = random.randint(0,3)
    inputs.append(possibilities[j])

def createLabel(input):
    x0 = input[0]
    x1 = input[1]
    return (x0 == 1 or x1 ==1) and not (x0==1 and x1==1)

def grdmse(data,network):
    scoreMatrix = np.empty((len(network.weights),len(network.weights[0])))
    MSE = network.mse(data)
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            tNetwork = copy.deepcopy(network)
            tNetwork.weights[i][j] += epsilon
            TMSE = tNetwork.mse(data)
            score = (TMSE-MSE)/epsilon
            scoreMatrix[i][j] = score
    return scoreMatrix

def sigmoid(x):
    return 1 / (1 + m.e ** -x)

class network():

    def __init__(self,inputSize,depth,width):
        trueWidth = max(depth,width)+1
        self.inputNodes = np.empty(inputSize+1)
        self.inputNodes[0] = 1
        self.hiddenNodes = np.empty((depth,width+1))
        self.hiddenNodes[:][0] = 1
        self.initialWeights(depth+1,trueWidth)

    def initialWeights(self,depth,width):
        #np.random.seed(329)
        self.weights = np.random.rand(depth, width)*2-1
        print(self.weights)

    def predict(self,input):
        for i in range(2):
            self.inputNodes[i+1] = input[i]
        self.calculateHiddenNodes()
        return self.calculateOutputNode()

    def calculateHiddenNodes(self):
        for i in range(len(self.hiddenNodes)):
            layer = copy.deepcopy(self.hiddenNodes[i])
            for j in range(1,len(layer)):
                layer[j] = 0
                if i == 0:
                    for k in range(len(self.inputNodes)):
                        layer[j] += self.weights[0][k] * self.inputNodes[k]
                    layer[j] = sigmoid(layer[j])
                else:
                    for k in range(len(self.hiddenNodes[j-1])):
                        layer[j] += self.weights[i][k] * self.hiddenNodes[j-1][k]
                    layer[j] = sigmoid(layer[j])
            self.hiddenNodes[i] = layer

    def calculateOutputNode(self):
        outputNode = 0
        #print(self.hiddenNodes[-1])
        for i in range(len(self.hiddenNodes[-1])):
            outputNode += self.weights[-1][i] * self.hiddenNodes[-1][i]
        outputNode = outputNode/len(self.hiddenNodes[-1])
        return outputNode

    def mse(self,inputs):
        SE = 0
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            target = createLabel(inputs[i])
            #print("pred: "+str(prediction))
            #print(prediction-target)
            SE += (prediction - target)**2
        MSE = SE/len(inputs)
        return MSE

    def changeWeights(self,scoreMatrix):
        oldW = copy.deepcopy(self.weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] - lr*scoreMatrix[i][j]
        print(self.weights)

    def test(self,inputs):
        mismatch = 0
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            label = createLabel(inputs[i])
            if((prediction - label) > 0.1):
                print(prediction)
                print(int(label))
                mismatch +=1
        print("Accuracy: "+str(100-mismatch/len(inputs)*100))
net = network(2,2,2)

scoreMatrix = 0
for epoch in range(epochs):
    OSM = scoreMatrix
    random.shuffle(inputs)
    scoreMatrix = grdmse(inputs,net)
    net.changeWeights(scoreMatrix)

net.test(inputs)








