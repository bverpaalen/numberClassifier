import numpy as np
import math as m


class xor_net():
    
    def __init__(self, x1 = 0, x2 = 1):
        self.x1 = x1
        self.x2 = x2
        self.initialWeights()
        
    def initialWeights(self):
        np.random.seed(200)
        self.weights = np.random.rand(3, 3)
    
class GradientD(xor_net):
    
    def mse(self, weights,data):
        for i in range(len(data)):
            actualData = data[0]
            desiredData = data[1]
            error = (self.predictActivations(actualData, weights) - desiredData)**2
        return np.mean(error)

    def actFunction(self,x):
        y = 1 / (1 + m.e ** -x)
#        y = np.tanh(x)[0]
#        y = np.maximum(0, x)
#        y = m.exp ^ x / sum(m.exp ^ x)
        return y

    def predictActivations(self, inputs, weights):

        y1_Activations = self.actFunction(weights[0][0] * inputs[0] + weights[0][1] * inputs[1]) + weights[0][2]
        y2_Activations = self.actFunction(weights[1][0] * inputs[0] + weights[1][1] * inputs[1]) + weights[1][2]

        prediction = self.actFunction(y1_Activations + y2_Activations)
        return prediction
    
    
    def grdmse(self, weights):
        delta = np.zeros((3, 3))
        dWeights = np.copy(weights)
        
        for i in range(0, 9):
            dWeights[m.trunc(i/3)][i%3] += epsilon
        
        delta[i] = (self.mse(dWeights) - self.mse(weights)) / epsilon
        return delta

#    def misClassify():


#Initialization
network = xor_net()

net = GradientD()

#Data :
x = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 1, 1, 0]

epsilon = 0.001
iterations = 1000
learningRate = [0.01, 0.1, 0.5, 1.0 , 2.5, 10]


#Gradient Descent Algorithm :

for l in learningRate:
    MSE = []
    misClassified = 0
    for i in range(iterations):
        for j in range(5):
            inputs = x[j]
            output = targets[j]
            data = [inputs, output]

            MSE.append(net.mse(network.weights, data))

            network.weights -= learningRate * net.grdmse(network.weights)
            prediction = GradientD.predictActivations(inputs,network.weights)
            if prediction != output:
                misClassified += 1

        

    
    print('The Mean Square Errors for the learning rate ' + l +' are : ' + MSE)
    print('The misclassified cases for the learning rate ' + l + ' are : ' + misClassified)