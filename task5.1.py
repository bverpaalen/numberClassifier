import numpy as np
import math as m
import copy

epochs = 1000
epsilon = 0.005
lr = 0.05

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
labels = [0, 1, 1, 0]


def grdmse(data,network):
    scoreMatrix = np.empty((len(network.weights),len(network.weights[0])))
    MSE = network.mse(data,labels)
    for i in range(len(network.weights)):
        for j in range(len(network.weights[i])):
            tNetwork = copy.deepcopy(network)
            tNetwork.weights[i][j] += epsilon
            TMSE = tNetwork.mse(data,labels)
            score = (TMSE-MSE)/epsilon
            scoreMatrix[i][j] = score
    return scoreMatrix

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
        self.weights = np.random.rand(depth, width)

    def predict(self,input):
        for i in range(2):
            self.inputNodes[i+1] = input[i]
        self.calculateHiddenNodes()
        return self.calculateOutputNode()

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
        print(outputNode)

        return outputNode

    def mse(self,inputs,labels):
        SE = 0
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            target = labels[i]

            SE += (prediction - target)**2
        MSE = np.mean(SE)
        print(MSE)
        return MSE

    def changeWeights(self,scoreMatrix):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = self.weights[i][j] - lr*scoreMatrix[i][j]

    def test(self,inputs,labels):
        mismatch = 0
        for i in range(len(inputs)):
            prediction = self.predict(inputs[i])
            if(prediction != labels[i]):
                mismatch +=1
        print(mismatch)
net = network(2,2,2)

for epoch in range(epochs):
    for i in range(len(inputs)):
        scoreMatrix = grdmse(inputs,net)
        net.changeWeights(scoreMatrix)

net.test(inputs, labels)








