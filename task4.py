import numpy as np



class Perceptron():
    
    #Contructor for initializing the object:
    def __init__(self, inputData, outputData, classes, dimensions):
        self.inputData = self.addBias(inputData)
        self.outputData = outputData
        self.weights = self.initialWeights(classes, dimensions)
        
    def addBias(self, inputData):
        a, b = np.shape(inputData)
        c = np.ones((a, 1))
        return np.hstack((c, inputData))
    
    def initialWeights(self, classes, dimensions):
        weights = np.random.rand(classes, dimensions)
        return np.array(weights)
    
    
    def backPropagate(self, weights, predictedOutput, outputData):
        count = 1
        iterations = 0

        while(count > 0 or iterations > 100):
            count = 0
            if outputData != predictedOutput:
                weights = self.updateWeight(weights, predictedOutput, outputData)
                count += 1
    
            iterations += 1
            return weights

    def updateWeight(self, weights, predictedOutput, outputData):
        
        for i in range (0, 257):
            
            if weights[i] < 0:
                weights[i] += 1
            else:
                weights[i] += 0.1
        
        return weights
    


trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

inputTraining = np.loadtxt(trainDataIn, dtype = 'float', delimiter = ',')
outputTraining = np.loadtxt(trainDataOut, dtype = 'float', delimiter = ',')

inputTest = np.loadtxt(testDataIn, dtype = 'str', delimiter = ',')
outputTest = np.loadtxt(testDataOut, dtype = 'str', delimiter = ',')


#Training on 1707 images in the training set of the MNIST data:
dim = 256 + 1 #the dimensions of a 16*16 sized image & 1 as a bias for the perceptron
cl = 10

Train = Perceptron(inputTraining, outputTraining, cl, dim)

#Initialization:
iTrain = Train.inputData
oTrain = Train.outputData

prediction = np.zeros(len(oTrain))
weight_matrix = Train.weights

for i in range (0, len(oTrain)):
    j = 0
    prediction[i] = np.argmax(np.dot((iTrain[i], weight_matrix[j])))
    j += 1
   
    if j > 257:
        j = 0
        

iteration = 0
misclassify = []

for i in range(0, len(oTrain)):
    x = int(oTrain[i])
    
    while (oTrain[i] != prediction[i] or iteration > 0):
        weight_matrix[x, :] = Train.backPropagate(weight_matrix[x, :], prediction[i], oTrain[i])
        
        if oTrain[i] != prediction[i]:
            misclassify.append(i)
        
        prediction[i] = Train.feedForward(iTrain[i])
    
    print('Unfortunately, the' + i + 'th image remains misclassified')



Test = Perceptron(inputTest, outputTest, cl, dim)

#Initialization:
iTest = Test.inputData
oTest = Test.outputData
weight_matrix2 = Test.weights

accuracy = 0

for inputD, outputD in zip(iTest, oTest):
    predictionTest = np.argmax(np.dot(inputD, weight_matrix2))
    
    if(outputD == predictionTest):
        accuracy += 1
accuracy /= inputTest.shape[0]
print('The accuracy of the algorthim is :' + str(accuracy * 100))