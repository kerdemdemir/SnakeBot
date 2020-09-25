import json
from fileinput import input

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
import datetime

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import TransactionHelper as transHelper
import DynamicTuner


transactionBinCountList = [6,8]
totalTimeCount = 6
isTrainCurves = True
totalUsedCurveCount = 4
isConcanateCsv = False
acceptedProbibilty = 0.9
testRatio = 4
transParamList = [inputManager.TransactionParam(250,  10),
                  inputManager.TransactionParam(500,  10),
                  inputManager.TransactionParam(750,  10),
                  inputManager.TransactionParam(1500,  10),
                  inputManager.TransactionParam(2500,  10)]

currentProbs = []


def ReadFileAndCreateReshaper( fileName ):
    print("Reading ", fileName )
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    reshaper = inputManager.ReShapeManager(transParamList)

    for jsonElem in jsonDictionary:
        reshaper.addANewCurrency(jsonElem,False)
    file.close()
    return  reshaper

def AddExtraToTuneShaper ( fileName, shaper):
    jsonDictionary = {}
    try:
        jsonDictionary = json.load(open(os.path.abspath(os.getcwd()) + "/Data/TuneData/" + fileName, "r"))
        for jsonElem in jsonDictionary:
            shaper.addANewCurrency(jsonElem, True)
    except:
        print("There was a exception in ", fileName)

def ReadFilesInTuneFolder( folderPath, reshaperTuner ):
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    for fileName in onlyfiles:
        print(" Reading for tuner: ", fileName)
        AddExtraToTuneShaper(fileName, reshaperTuner)

def AddExtraToShaper ( fileName, shaper, IsTransactionOnly):
    print("Reading ", fileName, " ", IsTransactionOnly)
    file = open(fileName, "r")
    jsonDictionary = {}
    try:
        jsonDictionary = json.load(file)
        for jsonElem in jsonDictionary:
            shaper.addANewCurrency(jsonElem, IsTransactionOnly)
    except:
        print("There was a exception in ", fileName)
    file.close()

def Predict ( messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, trainingReshaper):
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")
    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    resultStr = ""
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        extraStuff = resultsTransactionFloat[-transHelper.ExtraFeatureCount:]
        justTransactions = resultsTransactionFloat[:-transHelper.ExtraFeatureCount]
        currentTransactionList = DynamicTuner.MergeTransactions(justTransactions, transParam.msec, transParam.gramCount)
        scores = trainingReshaper.getScoreList(resultsChangeFloat)

        totalFeatures = currentTransactionList + extraStuff + resultsTimeFloat[-3:] + scores
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



onlyTransactions = ["learning_15_15_15.txt", "learning_42_05_05.txt", "learning_71_25_26.txt", "learning_87_18_19.txt" ]
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




print("All added now scores")
#trainingReshaper.transactionHelper.Print()
a = datetime.datetime.now()
trainingReshaper.assignScores()
b = datetime.datetime.now()
elapsedTime = b - a
print("Assigned scores ", elapsedTime.seconds)
sys.stdout.flush()

mlpTransactionList = []
mlpTransactionScalerList = []
for transactionIndex in range(len(transParamList)):
    transParam = transParamList[transactionIndex]
    numpyArr = trainingReshaper.toTransactionFeaturesNumpy(transactionIndex)
    mlpTransaction = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='relu',
                                                  solver='adam', max_iter=750)
    mlpTransactionList.append(mlpTransaction)
    transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
    mlpTransactionScalerList.append(transactionScaler)
    X = transactionScaler.transform(numpyArr)
    y = trainingReshaper.toTransactionResultsNumpy(transactionIndex)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

    mlpTransaction.fit(X_train, y_train)

    predict_test = mlpTransaction.predict_proba(X_test)
    finalResult = predict_test[:,1] >= acceptedProbibilty
    predict_test = np.delete(predict_test, 0 , 1 )

    print(" Transactions time: ", transParam.msec, " Transaction Index ", transParam.gramCount, "Index ", transactionIndex)
    print(confusion_matrix(y_test, finalResult))

trainingReshaper.ClearMemory()


currentProbs = [0.5, 0.5] + currentProbs
print( " Current tuned probibilities are : ", currentProbs)
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
        score = trainingReshaper.getScore(resultsChangeFloat)
        print( " Score for list: ", priceStrList, " is ", score)
        socket.send_string(str(score), encoding='ascii')
    elif command == "Adjust":
        print( " I will send back the new probabilities ", currentProbs)
        socket.send_string(str(currentProbs), encoding='ascii')

