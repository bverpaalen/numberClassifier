import numpy as np

dataPath = "./data/train_"
dataIn = dataPath+"in.csv"
dataOut = dataPath+"out.csv"

def main():
    i=0
    numberVectorDic = {}
    modelDic = {}
    inFile = open(dataIn)
    outFile = open(dataOut)

    #Fill dics
    for i in range(0,10):
        numberVectorDic.update({i:[]})
        modelDic.update({i:{}})

    GetNumberVectors(inFile,outFile,numberVectorDic)
    GetNumberAverageAndRadius(numberVectorDic, modelDic)
    print(modelDic)

def GetNumberVectors(inFile, outFile, numberVectorDic):
    #First read
    inLine = inFile.readline();
    outLine = outFile.readline();

    #When line is readable get vector/output
    while (inLine and outLine):
        output = int(outLine[0])
        vector = list(map(float, inLine.replace("\n", "").split(",")));

        numberVectorDic[output].append(vector)

        inLine = inFile.readline()
        outLine = outFile.readline();

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

main()