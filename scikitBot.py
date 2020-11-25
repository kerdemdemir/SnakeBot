import json
from fileinput import input

import ExtraDataManager as extraDataMan
import MarketStateManager as marketState
import zmq
import numpy as np
import sys
import os
import functools
from os import listdir
from os.path import isfile, join
import datetime
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import DynamicTuner
import SuddenChangeTransactions
import TransactionBasics

transactionBinCountList = [6,8]
totalTimeCount = 6
isTrainCurves = True
totalUsedCurveCount = 4
isConcanateCsv = False
acceptedProbibilty = 0.7
testRatio = 4
transParamList = [TransactionBasics.TransactionParam(1000,  5),
                  TransactionBasics.TransactionParam(2000,  5),
                  TransactionBasics.TransactionParam(4000,  5)]

currentProbs = []



def Predict ( messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, trainingReshaper):
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")
    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    resultStr = ""
    #scores = trainingReshaper.getScoreList(resultsChangeFloat)

    minMaxPriceRatio = transHelper.GetPeaksRatio(resultsChangeFloat, 7)
    #print(resultsChangeFloat, " ", resultsChangeFloat[7], " ", minMaxPriceRatio)
    if transHelper.ExtraPeakRatioCount != 0 and len(minMaxPriceRatio) != transHelper.ExtraPeakRatioCount:
        print("Bad extra data ", len(minMaxPriceRatio), " ", transHelper.ExtraPeakRatioCount)
        minMaxPriceRatio = minMaxPriceRatio[:transHelper.ExtraPeakRatioCount]

    #totalPerExtra = transHelper.ExtraPerDataInfo * len(transParamList)
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        extraCount = transHelper.ExtraFeatureCount+transHelper.ExtraLongPriceStateCount
        extraStuff = resultsTransactionFloat[-extraCount:]
        extraCount += len(transParamList) * transHelper.ExtraPerDataInfo
        justTransactions = resultsTransactionFloat[:-extraCount]
        currentTransactionList = DynamicTuner.MergeTransactions(justTransactions, transParam.msec, transParam.gramCount)

        perExtraStartIndex = -extraCount + transactionIndex * transHelper.ExtraPerDataInfo
        curExtra = resultsTransactionFloat[ perExtraStartIndex : perExtraStartIndex + transHelper.ExtraPerDataInfo]
        marketState = dynamicMarketState.curUpDowns
        totalFeatures = currentTransactionList + extraStuff + marketState + resultsTimeFloat[-3:] + minMaxPriceRatio
        totalFeaturesNumpy = np.array(totalFeatures).reshape(1, -1)
        totalFeaturesScaled = mlpTransactionScalerList[transactionIndex].transform(totalFeaturesNumpy)
        print("I will predict: ", totalFeatures, " scaled: ", totalFeaturesScaled)
        npTotalFeatures = np.array(totalFeaturesScaled)
        npTotalFeatures = npTotalFeatures.reshape(1, -1)
        predict_test = mlpTransactionList[transactionIndex].predict_proba(npTotalFeatures)
        curResultStr = str(predict_test) + ";"
        resultStr += curResultStr

    resultStr = resultStr[:-1]
    print("Results are: ", resultStr)
    return resultStr

def Learn():
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        numpyArr = suddenChangeManager.toTransactionFeaturesNumpy(transactionIndex)
        #numpyArr = extraDataManager.concanate(numpyArr,transactionIndex)


        mlpTransaction = MLPClassifier(hidden_layer_sizes=(24, 24, 24), activation='relu',
                                       solver='adam', max_iter=500)
        mlpTransactionList.append(mlpTransaction)
        transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
        mlpTransactionScalerList.append(transactionScaler)
        X = transactionScaler.transform(numpyArr)
        y = suddenChangeManager.toTransactionResultsNumpy(transactionIndex) #+ extraDataManager.getResult(transactionIndex)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
        #X_test = transactionScaler.transform(extraDataManager.getNumpy(transactionIndex))
        #y_test = extraDataManager.getConcanatedResult(transactionIndex)

        mlpTransaction.fit(X_train, y_train)

        predict_test = mlpTransaction.predict_proba(X_test)
        #print(predict_test)
        
        finalResult = predict_test[:, 1] >= 0.6
        print("60 ",confusion_matrix(y_test, finalResult))
        #print(predict_test)
        #predict_test = np.delete(finalResult, 0, 1)

        print(" Transactions time: ", transParam.msec, " Transaction Index ", transParam.gramCount, "Index ",
              transactionIndex)

        sys.stdout.flush()

dynamicMarketState = marketState.MarketStateManager()
extraFolderPath = os.path.abspath(os.getcwd()) + "/Data/ExtraData/"
extraDataManager = extraDataMan.ExtraDataManager(extraFolderPath,transParamList,dynamicMarketState)


suddenChangeManager = SuddenChangeTransactions.SuddenChangeManager(transParamList)

mlpTransactionList = []
mlpTransactionScalerList = []
Learn()

print("Memory cleaned")
sys.stdout.flush()
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/peakLearner")
while True:
    #  Wait for next request from client
    message = socket.recv_string(0, encoding='ascii')
    print("Received request: %s" % message)
    messageChangeTimeTransactionStrList = message.split(";")
    command = messageChangeTimeTransactionStrList[0]

    if command == "Predict":
        messageChangeTimeTransactionStrList = messageChangeTimeTransactionStrList[1:]
        resultStr = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, trainingReshaper)
        print("Results are: ", resultStr)
        #  Send reply back to client
        socket.send_string(resultStr, encoding='ascii')
        sys.stdout.flush()
    elif command == "GetScore":
        priceStrList = messageChangeTimeTransactionStrList[1].split(",")
        resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
        scoreList = trainingReshaper.getScoreList(resultsChangeFloat)
        print( " Score for list: ", priceStrList, " is ", scoreList)
        socket.send_string(str(scoreList), encoding='ascii')
    elif command == "Peak":
        print( " New peak will be added ")
        isBottom = messageChangeTimeTransactionStrList[1] == "Bottom"
        dynamicMarketState.addRecent(isBottom)
        socket.send_string("Done", encoding='ascii')

