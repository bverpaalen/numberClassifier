import numpy as np
import math as m


class xor_net():
    
    def __init__(self, x1 = 0, x2 = 1, weights):
        x1 = self.x1
        x2 = self.x2
        weightMatrix = self.intialWeights(weights)
        
    def initialWeights(self, weights):
        np.random.seed(200)
        weights = np.random.rand(3, 3)
        return weights
    
    
class GradientD(xor_net):
    
    def mse(self, weights):
        error = (self.predictActivation(actualData, weights) - desiredData)**2 for actualData, desiredData in data
        return np.mean(error)

    def actFunction(x):
        y = 1 / (1 + m.e ** -x)
#        y = np.tanh(x)[0]
#        y = np.maximum(0, x)
#        y = m.exp ^ x / sum(m.exp ^ x)
        return y

    def predictActivations(self, x, w):
        y1_Activations = self.actFunction( self.actFunction( w[0] * x[0] + w[2] * x[1] ) + w[4] )
        y2_Activations = self.actFunction( self.actFunction( w[1] * x[0] + w[3] * x[1] ) + w[5] )
        
        prediction = self.actFunction(y1_Activations + y2_Activations)
        return prediction
    
    
    def grdmse(self, weights):
        delta = np.zeros((3, 3))
        dWeights = np.copy(weights)
        
        for i in range(0, 9):
            dWeights[i] += epsilon
        
        delta[i] = (self.mse(dWeights) - self.mse(weights)) / epsilon
    return delta

    def misClassify():
        



#Initialization :
network = xor_net()
weights = network.weightMatrix

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
    
    for j in range(5):
        inputs = x[i]
        output = targets[i]
        data = np.concatenate(inputs, output)
        
        for i in range(iterations):
            weights -= learningRate * net.grdmse(weights)
            MSE.append(net.mse(weights))
        
            if prediction != output:
                misClassified += 1
    
    print('The Mean Square Errors for the learning rate ' + l +' are : ' + MSE)
    print('The misclassified cases for the learning rate ' + l + ' are : ' + misClassified)