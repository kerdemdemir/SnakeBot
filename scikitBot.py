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
        riseAndTimeStrList = peakData.split(",")
        if len(riseAndTimeStrList) < 2:
            continue
        riseAndTimeList = list(map(lambda x: input.RiseMinute(x), riseAndTimeStrList))
        reshaper.addLinePeaks(riseAndTimeList)
    file.close()
    return  reshaper


trainingReshaper = ReadFileAndCreateReshaper("C:\\Users\\Erdem\\Downloads\\learning.txt")
trainingReshaper.assignScores()

testReshaper = ReadFileAndCreateReshaper("C:\\Users\\Erdem\\Downloads\\learningNew.txt")


for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount):
    numpyArr = trainingReshaper.toFeaturesNumpy(binCount)

    X_train = numpyArr
    y_train = trainingReshaper.toResultsNumpy(binCount)

    X_test = testReshaper.toFeaturesNumpy(binCount)
    y_test = testReshaper.toResultsNumpy(binCount)

    mlp = MLPClassifier(hidden_layer_sizes=(binCount*2,binCount*2,binCount*2), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))