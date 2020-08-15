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
import DynamicTuner

smallestTime = 250
transactionBinCountList = [6,7]
totalTimeCount = 3
isTrainCurves = True
totalUsedCurveCount = 4
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

def AddExtraToTuneShaper ( fileName, shaper):
    jsonDictionary = {}
    try:
        jsonDictionary = json.load(open(os.path.abspath(os.getcwd()) + "/Data/TuneData/" + fileName, "r"))
        for jsonElem in jsonDictionary:
            shaper.addANewCurrency(jsonElem, True)
    except:
        file = open(os.path.abspath(os.getcwd()) + "/Data/TuneData/" + fileName, "r")
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
                            try:
                                jsonElem = json.loads( jsonStr )
                                shaper.addANewCurrency(jsonElem, True)
                            except:
                                print("Another exception")
                                continue
                        else:
                            isAlert = False
                    else:
                        curCount = 1
                        isAlert = True
                        print(isAlert, " ", jsonStr)

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
                            try:
                                jsonElem = json.loads( jsonStr )
                                shaper.addANewCurrency(jsonElem, IsTransactionOnly)
                            except:
                                print("Another exception")
                                continue
                        else:
                            isAlert = False
                    else:
                        curCount = 1
                        isAlert = True
                        print(isAlert, " ", jsonStr)



    file.close()

def Predict ( messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, mlpScalerList, mlpList, mixTransactionLearner ):
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")
    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    resultStr = ""
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        extraStuff = resultsTransactionFloat[-4:]
        justTransactions = resultsTransactionFloat[:-4]
        currentTransactionList = MergeTransactions(justTransactions, transParam.msec, transParam.gramCount)
        totalFeatures = currentTransactionList + extraStuff + [abs(resultsChangeFloat[-1]), resultsTimeFloat[-1]]
        totalFeaturesNumpy = np.array(totalFeatures).reshape(1, -1)
        totalFeaturesScaled = mlpTransactionScalerList[transactionIndex].transform(totalFeaturesNumpy)
        print("I will predict: ", totalFeatures, " scaled: ", totalFeaturesScaled)
        npTotalFeatures = np.array(totalFeaturesScaled)
        npTotalFeatures = npTotalFeatures.reshape(1, -1)
        predict_test = mlpTransactionList[transactionIndex].predict_proba(npTotalFeatures)
        curResultStr = str(predict_test) + ";"
        resultStr += curResultStr

    totalPredict = []
    for binCount in range(inputManager.ReShapeManager.maxFeatureCount - inputManager.ReShapeManager.minFeatureCount - 1):
        curCount = binCount + inputManager.ReShapeManager.minFeatureCount
        totalCurves = resultsChangeFloat[-curCount:] + resultsTimeFloat[-curCount:]
        npTotalCurves = np.array(totalCurves)
        npTotalCurves = npTotalCurves.reshape(1, -1)
        npTotalCurvesScaled = mlpScalerList[binCount].transform(npTotalCurves)
        print("I will predict the curves: ", totalCurves)
        predict_test = mlpList[binCount].predict_proba(npTotalCurvesScaled)
        curResultStr = str(predict_test) + ";"
        resultStr += curResultStr
        predict_test = np.delete(predict_test, 0, 1)
        if binCount < totalUsedCurveCount:
            totalPredict = np.append(totalPredict, predict_test)

    resultStr = resultStr[:-1]
    totalPredict = totalPredict.reshape(1, -1)
    print("I will predict the fusion: ", totalPredict)
    totalPredictResult = mixTransactionLearner.predict_proba(totalPredict)
    totalPredictResultStr = str(totalPredictResult) + ";"
    resultStr = totalPredictResultStr + resultStr
    print("Results are: ", resultStr)
    return resultStr



onlyTransactions = ["learning_15_15_15.txt"]
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
    mlpTransaction = MLPClassifier(hidden_layer_sizes=(transParam.gramCount+3, transParam.gramCount+3, transParam.gramCount+3), activation='relu',
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

resultPredicts = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - 1 - inputManager.ReShapeManager.minFeatureCount)]
mixTransactionLearner = MLPClassifier(hidden_layer_sizes=(4, 4, 4), activation='relu',
                                      solver='adam', max_iter=500)



del trainingReshaper


print("Start Tuning")
reshaperTuner = inputManager.ReShapeManager([inputManager.TransactionParam(125,80)])
ReadFilesInTuneFolder( os.path.abspath(os.getcwd()) + "/Data/TuneData/", reshaperTuner )

transactionTuner = DynamicTuner.PeakTransactionTurner(len(transParamList))
transactionTuner.Init(reshaperTuner, mlpTransactionScalerList, mlpTransactionList,transParamList)


if isTrainCurves:
    for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount-1):
        curIndex = binCount - inputManager.ReShapeManager.minFeatureCount
        numpyArr = reshaperTuner.toTransactionCurvesToNumpy(0, binCount)
        X = mlpScalerList[curIndex].transform(numpyArr)
        curResultPredict = mlpList[curIndex].predict_proba(X)
        resultPredicts[curIndex] = np.delete(curResultPredict, 0 , 1 )
        sys.stdout.flush()
    y = reshaperTuner.toTransactionPeakResultsNumpy(0)
    mergedArray = np.concatenate((resultPredicts[0], resultPredicts[1], resultPredicts[2], resultPredicts[3]), axis=1)
    X_trainMearged, X_testMerged, y_trainMerged, y_testMerged = train_test_split(mergedArray, y, test_size=0.2, random_state=40)
    print(" Transactions and curves merged: ")
    DynamicTuner.FitPredictAndPrint(mixTransactionLearner, X_trainMearged, X_testMerged, y_trainMerged, y_testMerged)
    print(" Tuning the curves ")
    finalResultNew = [0.0] * mergedArray.shape[1]
    DynamicTuner.AdjustTheBestCurve(mergedArray, y, finalResultNew)
    print( " Tuning result is ", finalResultNew)

sys.stdout.flush()

# print("Start Short memory tuning")
# shortMemReshaperTuner = inputManager.ReShapeManager([inputManager.TransactionParam(125,80)])
# AddExtraToTuneShaper("/Data/TuneData/learning_07_08.txt", shortMemReshaperTuner)
# transactionShortMemTuner = DynamicTuner.PeakTransactionTurner(len(transParamList))
# transactionShortMemTuner.Init(shortMemReshaperTuner, mlpTransactionScalerList, mlpTransactionList,transParamList)


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
        resultStr = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, mlpScalerList, mlpList, mixTransactionLearner)
        resultsTransactionFloat = [float(transactionResults[2:-2].split(" ")[1]) for transactionResults in resultStr.split(";")[1:len(transParamList) + 1]]
        print("Trasaction fusion will be asked: ", resultsTransactionFloat)
        tunerResult = transactionTuner.GetResult(resultsTransactionFloat)
        resultStr = tunerResult + ";" + resultStr

        # shortTunerResult = transactionShortMemTuner.GetResult(resultsTransactionFloat)
        # resultStr = shortTunerResult + ";" + resultStr

        print("Results are: ", resultStr)
        #  Send reply back to client
        socket.send_string(resultStr, encoding='ascii')
        sys.stdout.flush()
    elif command == "Train":
        isBottom = messageChangeTimeTransactionStrList[1] == "Bottom"
        messageChangeTimeTransactionStrList = messageChangeTimeTransactionStrList[2:]
        reJoinedMessageStr = ";".join(messageChangeTimeTransactionStrList)
        requestList = reJoinedMessageStr.split("|")
        isReTrain = False
        for request in requestList:
            print("Training predictions for : ", request )
            requestSplitedList = request.split(";")
            resultStr = Predict(requestSplitedList, mlpTransactionScalerList, mlpTransactionList, mlpScalerList, mlpList, mixTransactionLearner)
            isReTrain |= transactionTuner.Add(isBottom, resultStr)
        socket.send_string("Done", encoding='ascii')
        if isReTrain:
            transactionTuner.Train()
        sys.stdout.flush()

