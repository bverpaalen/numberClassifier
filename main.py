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

binSize = round(256/2)

#Creates prediction model using image vectors calculating an average model for each class, classes are (0..9)
def main():
    Assignment2()
    Assignment3(0,9)

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

    #PrintMatrix(confusionMatrix, "test")

def Assignment3(n1,n2):
    trainMatrix,testMatrix = CreateConfusionMatrixs(10)
    numberVectorDic = GetNumberVectors()
    classChance = CalculateClassChance()

    histograms = ExtractFeatures(numberVectorDic)

    #calculateAcurracy(histograms,classChance)

    for i in range(10):
        his1 = histograms[i][0]
        for j in range(i,10):
            his2 = histograms[j][0]
            answer = round(1-histogram_intersection_chance(his1,his2),1)
            trainMatrix[j][i] = answer
            trainMatrix[i][j] = answer
            print("compare "+str(i)+" and "+str(j)+": "+str(histogram_intersection_chance(his1,his2)))
    PrintMatrix(trainMatrix,"hisIntersection")

def calculateAcurracy(histograms,classChance,pathIn = trainDataIn,pathOut = trainDataOut):
    good = 0
    false = 0

    fileIn = open(pathIn)
    fileOut = open(pathOut)

    lineIn = fileIn.readline()
    lineOut = fileOut.readline()

    while(lineIn and lineOut):
        label = lineOut[0]
        vector = list(map(float,lineIn.replace("\n", "").split(",")))
        feature = extractFeature(vector)
        chances = [-1] * 10

        for key in histograms.keys():
            chance = chance_in_histogram(histograms[key],feature) * classChance[str(key)]
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

def histogram_intersection_chance(h1, h2):
    sm = 0
    for i in range(binSize-1):
        sm += min(h1[i],h2[i])
    return 2*sm/(sum(h1) + sum(h2))

def chance_in_histogram(hist,x):
    bins = hist[1]

    for i in range(0,len(bins)-1):
        min = bins[i]
        max = bins[i+1]

        if(x > min and x < max):
            inBin = hist[0][i]
            chance = sum(hist[0]) / inBin
            return chance
    return 0

def ExtractFeatures(numberVectorDic):
    features = {}
    for key in numberVectorDic.keys():
        keyVector = numberVectorDic[key]
        feature = extractFeatureVectors(keyVector)
        features.update({key: np.histogram(feature,bins=binSize)})
    return features


def extractFeatureVectors(keyVector):
    feature = []
    for i in range(len(keyVector)):
        vector = keyVector[i]
        toAppend = extractFeature(vector)
        feature.append(toAppend)
    return feature


def extractFeature(vector):
    toAppend = 0
    for j in range(len(vector)):

        if (vector[j] > 0.8 or vector[j] < -0.8):
            toAppend += 1
    return toAppend

def CreateConfusionMatrixs(size=10):
    matrix = {}

    for i in range(size):
        list = []
        for j in range(size):
            list.append(0)
        matrix.update({i:list})

    return matrix,matrix

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