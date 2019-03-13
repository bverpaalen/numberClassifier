import numpy as np

def histogram_intersection_chance(h1, h2,binSize):
    sm = 0
    for i in range(binSize-1):
        if(0 < h2[i]):
            sm += h1[i]
    return sm/sum(h1)

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

def ExtractFeatures(numberVectorDic,binSize):
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

        if (vector[j] < -0.9):
            toAppend += 1
    return toAppend