import math
import numpy as np
from sklearn import metrics as skm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

#Creates prediction model using image vectors calculating an average model for each class, classes are (0..9)
def main():
    Assignment2()
    Assignment3()

def Assignment2():
    modelDic = {}
    confusionMatrix,trainMatrix = CreateConfusionMatrixs(10)

    # Fill dics for numbers 0..9
    for i in range(0, 10):
        modelDic.update({i: {}})

    numberVectorDic = GetNumberVectors()

    GetNumberAverageAndRadius(numberVectorDic, modelDic)

    print("Train set")
    TestModel(modelDic, trainDataIn, trainDataOut, confusionMatrix=trainMatrix)

    # PrintMatrix(trainMatrix,"train")

    print("Test set")
    TestModel(modelDic, testDataIn, testDataOut, confusionMatrix)

    PrintMatrix(confusionMatrix, "test")

def Assignment3():
    trainMatrix,testMatrix = CreateConfusionMatrixs(10)
    numberVectorDic = GetNumberVectors()

    features = ExtractFeatures(numberVectorDic)
    print(features)


def CreateConfusionMatrixs(size=10):
    matrix = {}
    list = []
    for i in range(size):
        list.append(0)

    for i in range(size):
        matrix.update({i:list})

    return matrix,matrix

def ExtractFeatures(numberVectorDic):
    features = {}
    for key in numberVectorDic.keys():
        data = numberVectorDic[key]
        feature = []
        for i in range(len(data)):
            feature.append(256 + sum(data[i]))
        features.update({key:feature})
        plt.hist(feature,bins=range(0,260,10))
        plt.show()
    return features

#Retrieving all vectors for each number in given dic
def GetNumberVectors(dataIn = trainDataIn, dataOut = trainDataOut):
    numberVectorDic = {}

    for i in range(0, 10):
        numberVectorDic.update({i: []})

    inFile = open(dataIn)
    outFile = open(dataOut)

    #First read
    inLine = inFile.readline()
    outLine = outFile.readline()

    #When line is readable get vector/output
    while (inLine and outLine):
        output = int(outLine[0])
        vector = list(map(float, inLine.replace("\n", "").split(",")));

        numberVectorDic[output].append(vector)

        inLine = inFile.readline()
        outLine = outFile.readline()
    inFile.close()
    outFile.close()

    return numberVectorDic

#Calculates average and radius for each number from given numberVectorDic
def GetNumberAverageAndRadius(numberVectorDic, modelDic):
    for key in numberVectorDic.keys():
        vectors = numberVectorDic[key]
        average = np.average(vectors,axis=0).tolist()

        radius = CalculateRadius(average,vectors)
        modelDic.update({key:{"average":average,"radius":radius}})

#Calculate radius by finding the biggest difference between i in average and vectors
def CalculateRadius(average,vectors):
    radius = []

    for i in range(len(vectors[0])):
        radius.append(0)

    for vector in vectors:
        for i in range(len(vector)):
            difference = vector[i] - average[i]
            if abs(difference) > abs(radius[i]):
                radius[i] = abs(difference)
    return radius

#Tests given model on given data files
def TestModel(model,inFilePath,outFilePath,confusionMatrix):
    positive = 0
    negative = 0

    inFile = open(inFilePath)
    outFile = open(outFilePath)

    inLine = inFile.readline()
    outLine = outFile.readline()

    #Loops through both files, creates prediction for each line and checks if rightfully predicted
    while(inLine and outLine):
        output = int(outLine[0])
        vector = list(map(float, inLine.replace("\n", "").split(",")));
        prediction = PredictNumber(model, vector)

        #Checks if prediction was right
        if(prediction == output):
            positive +=1
        else:
            negative +=1
            if(confusionMatrix):
                confusionMatrix[prediction][output] += 1

        inLine = inFile.readline()
        outLine = outFile.readline()

    inFile.close()
    outFile.close()

    print("True Classified: "+str(positive))
    print("False Classified: "+str(negative))
    print()

#Creates prediction for given vector using given model
def PredictNumber(model, vector):
    nearestDistance = math.inf
    prediction = -1

    for key in model.keys():
        average = np.array(model[key]["average"])
        vector = np.array(vector)

        distance = skm.euclidean_distances(vector.reshape(1,-1),average.reshape(1,-1))
        #distance = skm.pairwise.cosine_distances(vector.reshape(1,-1),average.reshape(1,-1))
        #distance = skm.pairwise.manhattan_distances(vector.reshape(1,-1),average.reshape(1,-1))

        if distance < nearestDistance:
            nearestDistance = distance
            prediction = key

    return prediction

def PrintMatrix(matrix,filename):
    npMat = np.empty([10,10])
    for key in matrix.keys():
        #print(str(key) + ":"+str(matrix[key]))
        npMat[key] = np.array(matrix[key])

    print(npMat)
    ax = sns.heatmap(npMat,annot=True)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Label")
    ax.figure.savefig(filename)

if __name__ == '__main__':
    main()