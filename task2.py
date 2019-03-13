import math
from sklearn import metrics as skm
import numpy as np
import matplotlib.pyplot as plt

import VectorCalculations as VC
import Matrix as Mat

trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

#Creates prediction model using image vectors calculating an average model for each class, classes are (0..9)
def Assignment2():
    modelDic = {}
    confusionMatrix,trainMatrix = Mat.CreateConfusionMatrixs(10)

    # Fill dics for numbers 0..9
    for i in range(0, 10):
        modelDic.update({i: {}})

    numberVectorDic = VC.GetNumberVectors(testDataIn,testDataOut)

    VC.GetNumberAverageAndRadius(numberVectorDic, modelDic)

    print("Train set")
    TestModel(modelDic, trainDataIn, trainDataOut, confusionMatrix=trainMatrix)

    # PrintMatrix(trainMatrix,"train")

    print("Test set")
    TestModel(modelDic, testDataIn, testDataOut, confusionMatrix)

    #PrintMatrix(confusionMatrix, "test")

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

Assignment2()