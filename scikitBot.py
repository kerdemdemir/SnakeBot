import json
import Input as input
import InputManager as inputManager
import ExtraDataManager as extraDataMan
import zmq
import numpy as np
import sys
import os

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

transactionBinCount = 6
msecs = 1000
isTrainCurves = True
totalUsedCurveCount = 3

def ReadFileAndCreateReshaper( fileName ):
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    reshaper = inputManager.ReShapeManager()

    for jsonElem in jsonDictionary:
        reshaper.addANewCurrency(jsonElem,msecs,transactionBinCount,False)
    file.close()
    return  reshaper

def AddExtraToShaper ( fileName, shaper, IsTransactionOnly):
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    for jsonElem in jsonDictionary:
        shaper.addANewCurrency(jsonElem,msecs,transactionBinCount,IsTransactionOnly)
    file.close()

trainingReshaper = ReadFileAndCreateReshaper("learning_23_06.txt")
AddExtraToShaper("learning24_06.txt",trainingReshaper, True)
AddExtraToShaper("learning_25_06.txt",trainingReshaper, True)
AddExtraToShaper("learning_26_29.txt",trainingReshaper, True)
AddExtraToShaper("learning_29_30.txt",trainingReshaper, True)
AddExtraToShaper("learning_30.txt",trainingReshaper, True)
AddExtraToShaper("learning_30_1.txt",trainingReshaper, True)
AddExtraToShaper("learning_1_3.txt",trainingReshaper, True)
AddExtraToShaper("learning_3_7.txt",trainingReshaper, True)
AddExtraToShaper("learning_7_9.txt",trainingReshaper, True)
AddExtraToShaper("learning_9_10.txt",trainingReshaper, True)
AddExtraToShaper("learning_10_12.txt",trainingReshaper, False)
AddExtraToShaper("learning_13_14.txt",trainingReshaper, True)
AddExtraToShaper("learning_14_15.txt",trainingReshaper, True)
AddExtraToShaper("learning15_15.txt",trainingReshaper, True)
AddExtraToShaper("learning15_16.txt",trainingReshaper, True)
AddExtraToShaper("learning_16_17.txt",trainingReshaper, True)
AddExtraToShaper("learning_17_18.txt",trainingReshaper,True)
AddExtraToShaper("learning_18_19.txt",trainingReshaper,True)
AddExtraToShaper("learning_19_20.txt",trainingReshaper,True)
AddExtraToShaper("learning_20_21.txt",trainingReshaper,True)
AddExtraToShaper("learning_21_21.txt",trainingReshaper,True)
AddExtraToShaper("learning_21_21_2.txt",trainingReshaper,False)

extraDataManager = extraDataMan.ExtraDataManager( inputManager.ReShapeManager.minFeatureCount,
                                                  inputManager.ReShapeManager.maxFeatureCount,
                                                  transactionBinCount+3,
                                                  os.path.abspath(os.getcwd()) + "/Data")

print("All added now scores")
#trainingReshaper.transactionHelper.Print()
if isTrainCurves:
    trainingReshaper.assignScores()
print("Assigned scores")
sys.stdout.flush()

mlpList = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - inputManager.ReShapeManager.minFeatureCount)]
mlpScalerList = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - inputManager.ReShapeManager.minFeatureCount)]

if isTrainCurves :
    for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount):
        curIndex = binCount - inputManager.ReShapeManager.minFeatureCount
        numpyArr = trainingReshaper.toFeaturesNumpy(binCount)

        mlpScalerList[curIndex] = preprocessing.StandardScaler().fit(numpyArr)
        X = mlpScalerList[curIndex].transform(numpyArr)
        y = trainingReshaper.toResultsNumpy(binCount)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
        y_test = trainingReshaper.toTestResultNumpy(X_test,binCount)
        nodeSize = min(8, binCount*2)
        mlpList[curIndex] = MLPClassifier(hidden_layer_sizes=(nodeSize,nodeSize,nodeSize), activation='relu', solver='adam', max_iter=500)
        mlpList[curIndex].fit(X_train,y_train)
        predict_test = mlpList[curIndex].predict(X_test)
        print( " Curves : ")
        print( confusion_matrix(y_test,predict_test))
        sys.stdout.flush()

numpyArr = trainingReshaper.toTransactionFeaturesNumpy(transactionBinCount)
numpyArr = extraDataManager.ConcanateTransactions(numpyArr, transactionBinCount+5)
mlpTransaction = MLPClassifier(hidden_layer_sizes=(transactionBinCount+2, transactionBinCount+2, transactionBinCount+2), activation='relu',
                                              solver='adam', max_iter=500)

transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
X = transactionScaler.transform(numpyArr)
y = trainingReshaper.toTransactionResultsNumpy()
y = extraDataManager.ConcanateResults(y)
testCount = len(y)//4
print( "Test count is: ", testCount)
X_train = np.concatenate((X[:testCount,:], X[-testCount:,:]))
y_train = np.concatenate((y[:testCount], y[-testCount:]))
X_test = X[testCount:-testCount,:]
y_test = y[testCount:-testCount]
print(X_test)
print(y_test)
print(X_train)
print(y_train)

mlpTransaction.fit(X_train, y_train)

predict_test = mlpTransaction.predict_proba(X_test)
finalResult = predict_test[:,1] >= 0.8
predict_test = np.delete(predict_test, 0 , 1 )
print(" Transactions : ", predict_test)
print(confusion_matrix(y_test, finalResult))

resultPredicts = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - 1 - inputManager.ReShapeManager.minFeatureCount)]
if isTrainCurves:
    for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount-1):
        curIndex = binCount - inputManager.ReShapeManager.minFeatureCount
        numpyArr = trainingReshaper.toTransactionCurvesToNumpy(binCount)
        numpyArr = extraDataManager.ConcanateFeature(numpyArr,binCount)
        X = mlpScalerList[curIndex].transform(numpyArr)
        X_test = X[testCount:-testCount,:]
        curResultPredict = mlpList[curIndex].predict_proba(X_test)
        resultPredicts[curIndex] = np.delete(curResultPredict, 0 , 1 )
        print( " Transaction Curves Bin Count: ", binCount, " Results: ", curResultPredict)
        sys.stdout.flush()



mergedArray = np.concatenate((predict_test, resultPredicts[0], resultPredicts[1], resultPredicts[2]), axis=1)
print(mergedArray)
X_trainMearged, X_testMerged, y_trainMerged, y_testMerged = train_test_split(mergedArray, y_test, test_size=0.1, random_state=40)
mixTransactionLearner = MLPClassifier(hidden_layer_sizes=(transactionBinCount, transactionBinCount, transactionBinCount), activation='relu',
                                              solver='adam', max_iter=500)
mixTransactionLearner.fit(X_trainMearged, y_trainMerged)
predict_test = mixTransactionLearner.predict(X_testMerged)
print(" Transactions and curves merged: ")
print(confusion_matrix(y_testMerged, predict_test))


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/peakLearner")

while True:
    #  Wait for next request from client
    message = socket.recv_string(0, encoding='ascii')
    print("Received request: %s" % message)
    messageChangeTimeTransactionStrList = message.split(";")
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")

    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    resultStr = ""

    totalFeatures = resultsTransactionFloat[:transactionBinCount + 3] + [resultsChangeFloat[-1], resultsTimeFloat[-1]]
    totalFeaturesNumpy = np.array(totalFeatures).reshape(1, -1)
    totalFeaturesScaled = transactionScaler.transform(totalFeaturesNumpy)
    print("I will predict: ", totalFeatures, " scaled: ", totalFeaturesScaled )
    npTotalFeatures = np.array(totalFeaturesScaled)
    npTotalFeatures = npTotalFeatures.reshape(1, -1)
    predict_test = mlpTransaction.predict_proba(npTotalFeatures)
    curResultStr = str(predict_test) + ";"
    resultStr += curResultStr
    totalPredict = np.delete(predict_test, 0 , 1 )

    for binCount in range (inputManager.ReShapeManager.maxFeatureCount-inputManager.ReShapeManager.minFeatureCount-1):
        curCount = binCount + inputManager.ReShapeManager.minFeatureCount
        totalCurves = resultsChangeFloat[-curCount:] + resultsTimeFloat[-curCount:]
        npTotalCurves = np.array(totalCurves)
        npTotalCurves = npTotalCurves.reshape(1,-1)
        npTotalCurvesScaled = mlpScalerList[binCount].transform(npTotalCurves)
        print("I will predict the curves: ", totalCurves)
        predict_test = mlpList[binCount].predict_proba(npTotalCurvesScaled)
        curResultStr = str(predict_test) + ";"
        resultStr += curResultStr
        predict_test = np.delete(predict_test, 0, 1)
        if binCount < totalUsedCurveCount:
            totalPredict = np.append(totalPredict,predict_test)

    resultStr = resultStr[:-1]
    totalPredict = totalPredict.reshape(1, -1)
    totalPredictResult = mixTransactionLearner.predict_proba(totalPredict)

    totalPredictResultStr = str(totalPredictResult) + ";"
    resultStr = totalPredictResultStr + resultStr
    print("Results are: " , resultStr)

    #  Send reply back to client
    socket.send_string(resultStr, encoding='ascii')
