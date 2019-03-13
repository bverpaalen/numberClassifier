import VectorCalculations as VC
import Matrix as mat
import Util as ut
import numpy as np

trainDataPath = "./data/train_"
trainDataIn = trainDataPath + "in.csv"
trainDataOut = trainDataPath + "out.csv"

testDataPath = "./data/test_"
testDataIn = testDataPath + "in.csv"
testDataOut = testDataPath + "out.csv"

binSize = round(256)

def Assignment3(n1,n2):
    trainMatrix,testMatrix = mat.CreateConfusionMatrixs(10)
    numberVectorDic = VC.GetNumberVectors(testDataIn,testDataOut)
    classChance = CalculateClassChance()

    histograms = ut.ExtractFeatures(numberVectorDic,binSize)

    #calculateAcurracy(histograms,classChance)

    for i in range(10):
        his1 = histograms[i][0]
        for j in range(i,10):
            his2 = histograms[j][0]
            trainMatrix[j][i] = round(ut.histogram_intersection_chance(his1,his2,binSize),1)
            trainMatrix[i][j] = round(ut.histogram_intersection_chance(his2,his1,binSize),1)
            print("compare "+str(i)+" and "+str(j)+": "+str(ut.histogram_intersection_chance(his1,his2,binSize)))
    mat.PrintMatrix(trainMatrix,"hisIntersection")

def calculateAcurracy(histograms,classChance,pathIn,pathOut):
    good = 0
    false = 0

    fileIn = open(pathIn)
    fileOut = open(pathOut)

    lineIn = fileIn.readline()
    lineOut = fileOut.readline()

    while(lineIn and lineOut):
        label = lineOut[0]
        vector = list(map(float,lineIn.replace("\n", "").split(",")))
        feature = ut.extractFeature(vector)
        chances = [-1] * 10

        for key in histograms.keys():
            chance = ut.chance_in_histogram(histograms[key],feature) * classChance[str(key)]
            chances[key] = chance
        sumChance = sum(chances)

        highest = -1
        if(sumChance > 0):
            for i in range(10):
                finalChance = chances[i] / sumChance
                if(finalChance > highest):
                    highest = finalChance
                    prediction = i
        else:
            print("No prediction")
            print(label)
            print(vector)
            prediction = -1

        if(str(prediction) == label):
            print("Correct label")
            good += 1
        else:
            print("Prediction: "+str(prediction)+", Label: "+str(label))
            false +=1
        lineIn = fileIn.readline()
        lineOut = fileOut.readline()

    print(good)
    print(false)
def CalculateClassChance(path=trainDataOut):
    file = open(path)
    numbers = {}
    total = 0
    line = file.readline()
    while(line):
        number = line[0]

        if number in numbers.keys():
            numbers[number] += 1
        else:
            numbers.update({number:1})
        total += 1
        line = file.readline()

    for number in numbers.keys():
        numbers[number] = numbers[number] / total
    return numbers

Assignment3(0,9)