import numpy as np

epochs = 20

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
            for j in range(0,10):
                if(j==label):
                    self.weights[i][j] = self.weights[i][j] + node
                else:
                    self.weights[i][j] = self.weights[i][j] - node

    def predict(self,input):
        dot = np.dot(input,self.weights)
        prediction = np.argmax(dot)
        return prediction

    def test(self,inputData,outputData):
        accuracy = 0
        i=0
        for inputD, outputD in zip(inputData, outputData):
            label = int(outputD)
            predictionTest = Train.predict(inputD)

            if(outputD == predictionTest):
                accuracy += 1
            #else:
            #    print('Unfortunately, the ' + str(i) + 'th image remains misclassified')
            i+=1
        accuracy /= inputData.shape[0]
        print('The accuracy of the algorthim is :' + str(accuracy * 100))
       
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

prediction = np.zeros(len(Train.outputData))
weight_matrix = Train.weights

iteration = 0
misclassify = 0
j = 0
while (iteration < epochs):
    iteration += 1
    for i in range(0, len(Train.outputData)):
        label = int(Train.outputData[i])
        prediction[i] = Train.predict(Train.inputData[i])

        if(prediction[i] != label):
            Train.backPropagate(label,Train.inputData[i],prediction[i])

        #Re-prediction by the perceptron:

Test = Perceptron(inputTest, outputTest, cl, dim)

Train.test(outputData=Train.outputData,inputData=Train.inputData)

#Testing on the 1000 images in the training set of the MNIST data:
iTest = Train.addBias(inputTest)
oTest = Test.outputData

Train.test(outputData = oTest,inputData = iTest)
