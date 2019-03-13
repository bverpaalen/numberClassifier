import numpy as np

#Retrieving all vectors for each number in given dic
def GetNumberVectors(dataIn, dataOut):
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