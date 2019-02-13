import math
import numpy as np

trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

def main():
    numberVectorDic = {}
    modelDic = {}
    inFile = open(trainDataIn)
    outFile = open(trainDataOut)

    #Fill dics
    for i in range(0,10):
        numberVectorDic.update({i:[]})
        modelDic.update({i:{}})

    GetNumberVectors(inFile,outFile,numberVectorDic)
    inFile.close()
    outFile.close()

    GetNumberAverageAndRadius(numberVectorDic, modelDic)

    print("Train set")
    TestModel(modelDic,trainDataIn,trainDataOut)

    print("Test set")
    TestModel(modelDic,testDataIn,testDataOut)

def GetNumberVectors(inFile, outFile, numberVectorDic):
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

def GetNumberAverageAndRadius(numberVectorDic, modelDic):
    #for each number (0..9) get average vector and calculate radius
    for key in numberVectorDic.keys():
        vectors = numberVectorDic[key]
        average = np.average(vectors,axis=0).tolist()

        radius = CalculateRadius(average,vectors)
        modelDic.update({key:{"average":average,"radius":radius}})

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

def TestModel(model,inFilePath,outFilePath):
    positive = 0
    negative = 0

    inFile = open(inFilePath)
    outFile = open(outFilePath)

    inLine = inFile.readline()
    outLine = outFile.readline()

    while(inLine and outLine):
        output = int(outLine[0])
        vector = list(map(float, inLine.replace("\n", "").split(",")));
        prediction = PredictNumber(model, vector)

        if(prediction == output):
            positive +=1
        else:
            negative +=1

        inLine = inFile.readline()
        outLine = outFile.readline()

    inFile.close()
    outFile.close()

    print("True Classified: "+str(positive))
    print("False Classified: "+str(negative))
    print()

def PredictNumber(model, vector):
    nearestDistance = math.inf
    prediction = -1

    for key in model.keys():
        average = np.array(model[key]["average"])
        vector = np.array(vector)
        distance = sum(np.abs(vector-average))

        if distance < nearestDistance:
            nearestDistance = distance
            prediction = key
    return prediction

main()