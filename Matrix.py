import numpy as np
import seaborn as sns
sns.set()

def CreateConfusionMatrixs(size=10):
    matrix = {}

    for i in range(size):
        list = []
        for j in range(size):
            list.append(0)
        matrix.update({i:list})

    return matrix,matrix

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