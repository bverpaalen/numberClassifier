import numpy as np

epochs = 100
learningRate = 0.05

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
    
    def initialWeights(self, numClasses, numInputs):
        self.weights = np.random.rand(numInputs, numClasses)
        return self.weights

    def backPropagate(self,label,input,prediction):
        
        for i in range (0, 257):
            node = input[i]
            nodePrediction = np.argmax(node*self.weights[i])

            if int(label) == int(nodePrediction):
                self.weights[i] = self.weights[i] + node
            else:
                self.weights[i] = self.weights[i] - node

    def predict(self,input):
        dot = np.dot(input,self.weights)
        prediction = np.argmax(dot)
        return prediction


trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

inputTraining = np.loadtxt(trainDataIn, dtype = 'float', delimiter = ',')
outputTraining = np.loadtxt(trainDataOut, dtype = 'float', delimiter = ',')

inputTest = np.loadtxt(testDataIn, dtype = 'float', delimiter = ',')
outputTest = np.loadtxt(testDataOut, dtype = 'float', delimiter = ',')


#Training on 1707 images in the training set of the MNIST data:
dim = 256 + 1 #the dimensions of a 16*16 sized image & 1 as a bias for the perceptron
cl = 10

Train = Perceptron(inputTraining, outputTraining, cl, dim)

#Initialization:
iTrain = Train.inputData
oTrain = Train.outputData

prediction = np.zeros(len(oTrain))
weight_matrix = Train.weights

#Prediction by the perceptron:
for i in range (0, len(oTrain)):
    label = int(oTrain[i])
    prediction[i] = Train.predict(iTrain[i])

iteration = 0
misclassify = 0
j = 0
while (iteration < epochs):
    iteration += 1
    for i in range(0, len(oTrain)):
        label = int(oTrain[i])
    
        Train.backPropagate(label,iTrain[i],prediction[i])

        #Re-prediction by the perceptron:
        prediction[i] = Train.predict(iTrain[i])


Test = Perceptron(inputTest, outputTest, cl, dim)

#Testing on the 1000 images in the training set of the MNIST data:
iTest = Test.inputData
oTest = Test.outputData

accuracy = 0
i=0
for inputD, outputD in zip(iTest, oTest):
    label = int(outputD)
    predictionTest = Train.predict(inputD)
    
    if(outputD == predictionTest):
        accuracy += 1

    print('Unfortunately, the ' + str(i) + 'th image remains misclassified')
    i+=1

accuracy /= inputTest.shape[0]
print('The accuracy of the algorthim is :' + str(accuracy * 100))