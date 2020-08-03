import json
import Input as input
import InputManager as inputManager
import ExtraDataManager as extraDataMan
import zmq
import numpy as np
import sys
import os
import functools
from os import listdir
from os.path import isfile, join


from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import TransactionHelper as transHelper

smallestTime = 125
transactionBinCountList = [6,8]
totalTimeCount = 5
isTrainCurves = True
totalUsedCurveCount = 3
isConcanateCsv = False
acceptedProbibilty = 0.9
testRatio = 4
transParamList = []

def MergeTransactions ( transactionList, msec, transactionBinCount ):
    index = msec // smallestTime
    totalElement = index * transactionBinCount
    arrayList = np.array_split(transactionList[-totalElement:], transactionBinCount)
    print(arrayList)
    mergeArray = list(map(lambda x: x.sum(), arrayList))
    summedArray = list(map(lambda x: transHelper.NormalizeTransactionCount(x), mergeArray))
    return summedArray

def ReadFileAndCreateReshaper( fileName ):
    print("Reading ", fileName )
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    for transactionBinCount in transactionBinCountList:
        for index in range(totalTimeCount):
            transactionParam = inputManager.TransactionParam(smallestTime * (index * 2 + 1), transactionBinCount)
            transParamList.append(transactionParam)
    reshaper = inputManager.ReShapeManager(transParamList)

    for jsonElem in jsonDictionary:
        reshaper.addANewCurrency(jsonElem,False)
    file.close()
    return  reshaper

def AddExtraToShaper ( fileName, shaper, IsTransactionOnly):
    print("Reading ", fileName, " ", IsTransactionOnly)
    file = open(fileName, "r")
    jsonDictionary = {}
    try:
        jsonDictionary = json.load(file)
        for jsonElem in jsonDictionary:
            shaper.addANewCurrency(jsonElem, IsTransactionOnly)
    except:
        file = open(fileName, "r")
        temp = file.readline()
        startIndex = 0
        curCount = 0
        isAlert = False
        for index in range(len(temp)):
            if temp[index] == "{":
                curCount += 1
                if curCount == 1:
                    startIndex = index
            elif temp[index] == "}" :
                curCount -= 1
                if curCount == 0 :
                    jsonStr = temp[startIndex:index + 1]
                    if  temp[index - 1] == "]":
                        #print(jsonStr)
                        if not isAlert:
                            jsonElem = json.loads( jsonStr )
                            shaper.addANewCurrency(jsonElem, IsTransactionOnly)
                        else:
                            isAlert = False
                    else:
                        curCount = 1
                        isAlert = True
                        print(isAlert, " ", jsonStr)



    file.close()


onlyTransactions = ["learning_12_10_12.txt"]
folderPath = os.path.abspath(os.getcwd()) + "/Data/CompleteData/"
onlyTransactions = list(map( lambda x:  folderPath+x, onlyTransactions))


onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
def compareInt(x,y):
    return int(x.split("_")[1]) - int(y.split("_")[1])

onlyfiles = list(sorted( onlyfiles, key=functools.cmp_to_key(compareInt) ))

onlyfiles = list(map( lambda x:  folderPath+x, onlyfiles))
trainingReshaper = ReadFileAndCreateReshaper(onlyfiles[0])
for fileName in onlyfiles:
    if fileName == onlyfiles[0]:
        continue
    elif fileName == onlyfiles[-1]:
        AddExtraToShaper(fileName, trainingReshaper, False)
    elif fileName in onlyTransactions:
        AddExtraToShaper(fileName, trainingReshaper, False)
    else:
        AddExtraToShaper(fileName, trainingReshaper, True)

if isConcanateCsv:
    extraDataManager = extraDataMan.ExtraDataManager( inputManager.ReShapeManager.minFeatureCount,
                                                  inputManager.ReShapeManager.maxFeatureCount,
                                                  9,
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

mlpTransactionList = []
mlpTransactionScalerList = []
for transactionIndex in range(len(transParamList)):
    transParam = transParamList[transactionIndex]
    numpyArr = trainingReshaper.toTransactionFeaturesNumpy(transactionIndex)
    if isConcanateCsv:
        numpyArr = extraDataManager.ConcanateTransactions(numpyArr, 6+5)
    mlpTransaction = MLPClassifier(hidden_layer_sizes=(transParam.gramCount+3, transParam.gramCount+3, transParam.gramCount+3), activation='relu',
                                                  solver='adam', max_iter=750)
    mlpTransactionList.append(mlpTransaction)
    transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
    mlpTransactionScalerList.append(transactionScaler)
    X = transactionScaler.transform(numpyArr)
    y = trainingReshaper.toTransactionResultsNumpy(transactionIndex)
    if isConcanateCsv:
        y = extraDataManager.ConcanateResults(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

    mlpTransaction.fit(X_train, y_train)

    predict_test = mlpTransaction.predict_proba(X_test)
    finalResult = predict_test[:,1] >= acceptedProbibilty
    predict_test = np.delete(predict_test, 0 , 1 )
    print(" Transactions time: ", transParam.msec, " Transaction Index ", transParam.gramCount, "Index ", transactionIndex)
    print(confusion_matrix(y_test, finalResult))

resultPredicts = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - 1 - inputManager.ReShapeManager.minFeatureCount)]
if isTrainCurves:
    for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount-1):
        curIndex = binCount - inputManager.ReShapeManager.minFeatureCount
        numpyArr = trainingReshaper.toTransactionCurvesToNumpy(2, binCount)
        if isConcanateCsv:
            numpyArr = extraDataManager.ConcanateFeature(numpyArr,binCount)
        X = mlpScalerList[curIndex].transform(numpyArr)
        curResultPredict = mlpList[curIndex].predict_proba(X)
        resultPredicts[curIndex] = np.delete(curResultPredict, 0 , 1 )
        print( " Transaction Curves Bin Count: ", binCount, " Results: ", curResultPredict)
        sys.stdout.flush()
    y = trainingReshaper.toTransactionResultsNumpy(2)


    mergedArray = np.concatenate((resultPredicts[0], resultPredicts[1], resultPredicts[2]), axis=1)
    print(mergedArray)
    X_trainMearged, X_testMerged, y_trainMerged, y_testMerged = train_test_split(mergedArray, y, test_size=0.2, random_state=40)
    mixTransactionLearner = MLPClassifier(hidden_layer_sizes=(4, 4, 4), activation='relu',
                                                  solver='adam', max_iter=500)
    mixTransactionLearner.fit(X_trainMearged, y_trainMerged)
    predict_test = mixTransactionLearner.predict(X_testMerged)
    print(" Transactions and curves merged: ")
    print(confusion_matrix(y_testMerged, predict_test))

del trainingReshaper


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/peakLearner")
transactionTuner = inputManager.PeakTransactionTurner(len(transParamList))
while True:
    #  Wait for next request from client
    message = socket.recv_string(0, encoding='ascii')
    print("Received request: %s" % message)
    messageChangeTimeTransactionStrList = message.split(";")
    command = messageChangeTimeTransactionStrList[0]

    if command == "Predict":
        priceStrList = messageChangeTimeTransactionStrList[1].split(",")
        timeStrList = messageChangeTimeTransactionStrList[2].split(",")
        transactionStrList = messageChangeTimeTransactionStrList[3].split(",")

        resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
        resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
        resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

        resultStr = ""
        for transactionIndex in range(len(transParamList)):
            transParam = transParamList[transactionIndex]
            extraStuff = resultsTransactionFloat[-4:]
            justTransactions = resultsTransactionFloat[:-4]
            currentTransactionList = MergeTransactions( justTransactions, transParam.msec, transParam.gramCount)
            totalFeatures = currentTransactionList + extraStuff + [abs(resultsChangeFloat[-1]), resultsTimeFloat[-1]]
            totalFeaturesNumpy = np.array(totalFeatures).reshape(1, -1)
            totalFeaturesScaled = mlpTransactionScalerList[transactionIndex].transform(totalFeaturesNumpy)
            print("I will predict: ", totalFeatures, " scaled: ", totalFeaturesScaled )
            npTotalFeatures = np.array(totalFeaturesScaled)
            npTotalFeatures = npTotalFeatures.reshape(1, -1)
            predict_test = mlpTransactionList[transactionIndex].predict_proba(npTotalFeatures)
            curResultStr = str(predict_test) + ";"
            resultStr += curResultStr
        totalPredict = []

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
    elif command == "Train":
        valueAndTime = messageChangeTimeTransactionStrList[1].split(",")
        valueAndTime = list(map( lambda x: float(x), valueAndTime))
        isBottom = valueAndTime[0] < 0.0
        jsonPeak = json.loads(messageChangeTimeTransactionStrList[2])
        resultStr = ""
        for transactionIndex in range(len(transParamList)):
            transParam = transParamList[transactionIndex]
            transPeakTemp = transHelper.TransactionPeakHelper( jsonPeak, transParam.msec, isBottom, valueAndTime[0], valueAndTime[1], None, None)
            transPeakTemp.AssignScores(transParam.gramCount)
            transactionPatterns = transPeakTemp.GetTransactionPatterns()
            curResultList = []
            for transactionPattern in transactionPatterns:
                totalFeatures = transactionPattern + [abs(valueAndTime[0]), valueAndTime[1]]
                totalFeaturesScaled = mlpTransactionScalerList[transactionIndex].transform(totalFeaturesNumpy)
                print("I will predict for training: ", totalFeatures, " scaled: ", totalFeaturesScaled)
                npTotalFeatures = np.array(totalFeaturesScaled)
                npTotalFeatures = npTotalFeatures.reshape(1, -1)
                predict_test = mlpTransactionList[transactionIndex].predict_proba(npTotalFeatures)
                curResult = predict_test[0][1]

                print("Result after after new peak for: ", transParam.msec, " ", transParam.gramCount, " is ", curResult)
            print("New peak result ", transParam)
            resultStr = str(transParam.goodResults) + ";" + str(transParam.badResults) + "|"
        resultStr = resultStr[:-1]
        print("Final result is ", resultStr)
        socket.send_string(resultStr, encoding='ascii')

