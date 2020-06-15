import json
import Input as input
import InputManager as inputManager
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix


def ReadFileAndCreateReshaper( fileName ):
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    reshaper = inputManager.ReShapeManager()

    for jsonElem in jsonDictionary:
        peakData = jsonElem["peak"]
        transactionData = jsonElem["transactionList"]

        riseAndTimeStrList = peakData.split(",")
        if len(riseAndTimeStrList) < 2:
            continue

        riseAndTimeList = list(map(lambda x: input.RiseMinute(x), riseAndTimeStrList))
        reshaper.addLinePeaks(riseAndTimeList)
        reshaper.addTransactions(transactionData)
    file.close()
    return  reshaper


trainingReshaper = ReadFileAndCreateReshaper("C:\\Users\\Erdem\\Downloads\\learningNew.txt")
trainingReshaper.assignScores()


for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount):
    numpyArr = trainingReshaper.toFeaturesNumpy(binCount)

    X = numpyArr
    y = trainingReshaper.toResultsNumpy(binCount)

    #print( X.shape, " ", y.shape,  X.shape, " ", y_1.shape,  X_2.shape, " ", y_2.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

    mlp = MLPClassifier(hidden_layer_sizes=(binCount*2,binCount*2,binCount*2), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print("bin count:", binCount, " ", confusion_matrix(y_train,predict_train))
    print("bin count:", binCount, " ", classification_report(y_train,predict_train))