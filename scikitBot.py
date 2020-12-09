import ExtraDataManager as extraDataMan
import MarketStateManager as marketState
import zmq
import numpy as np
import sys
import os

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import DynamicTuner
import SuddenChangeTransactions
import TransactionBasics
import PeakTransactions

transactionBinCountList = [6,8]
totalTimeCount = 6
isUsePeaks = False
totalUsedCurveCount = 4
isUseExtraData = True
acceptedProbibilty = 0.7
testRatio = 4
transParamList = [TransactionBasics.TransactionParam(1000, 10)]

currentProbs = []



def Predict ( messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList):
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")
    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    marketStateList = dynamicMarketState.curUpDowns
    resultStr = ""

    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        justTransactions = resultsTransactionFloat
        multipliedGramCount = TransactionBasics.GetTotalPatternCount(transParam.gramCount)
        currentTransactionList = DynamicTuner.MergeTransactions(justTransactions, transParam.msec, multipliedGramCount)
        basicList = TransactionBasics.CreateTransactionList(currentTransactionList)
        basicList = TransactionBasics.ReduceToNGrams(basicList, transParam.gramCount)
        currentTransactionList = TransactionBasics.GetListFromBasicTransData(basicList)
        # + marketStateList market state is cancelled for now
        #totalFeatures = currentTransactionList  + resultsChangeFloat[-TransactionBasics.PeakFeatureCount:] + resultsTimeFloat[-TransactionBasics.PeakFeatureCount:]
        if TransactionBasics.PeakFeatureCount == 0 :
            totalFeatures = currentTransactionList + marketStateList
        else:
            totalFeatures = currentTransactionList + marketStateList + resultsChangeFloat[-TransactionBasics.PeakFeatureCount:] + resultsTimeFloat[-TransactionBasics.PeakFeatureCount:]


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
        if isUsePeaks:
            numpyArrPeak = peakManager.toTransactionFeaturesNumpy(transactionIndex)
            numpyArr = np.concatenate((numpyArr, numpyArrPeak), axis=0)


        mlpTransaction = MLPClassifier(hidden_layer_sizes=(36, 36, 36), activation='relu',
                                       solver='adam', max_iter=500)
        mlpTransactionList.append(mlpTransaction)
        transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
        mlpTransactionScalerList.append(transactionScaler)
        X = transactionScaler.transform(numpyArr)
        y = suddenChangeManager.toTransactionResultsNumpy(transactionIndex) #+ extraDataManager.getResult(transactionIndex)
        if isUsePeaks:
            yPeak = peakManager.toTransactionResultsNumpy(transactionIndex)
            y += yPeak

        if isUseExtraData:
            X_test = transactionScaler.transform(extraDataManager.getNumpy(transactionIndex))
            y_test = extraDataManager.getConcanatedResult(transactionIndex)
            X_train = X
            y_train = y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

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
suddenChangeManager = SuddenChangeTransactions.SuddenChangeManager(transParamList)
if isUsePeaks:
    peakManager = PeakTransactions.PeakManager(transParamList)
if isUseExtraData:
    extraFolderPath = os.path.abspath(os.getcwd()) + "/Data/ExtraData/"
    extraDataManager = extraDataMan.ExtraDataManager(extraFolderPath,transParamList,suddenChangeManager.marketState)



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
        resultStr = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList)
        print("Results are: ", resultStr)
        #  Send reply back to client
        socket.send_string(resultStr, encoding='ascii')
        sys.stdout.flush()
    elif command == "Peak":
        print( " New peak will be added ")
        isRising = messageChangeTimeTransactionStrList[1] == "Increase"
        dynamicMarketState.addRecent(isRising)
        socket.send_string("Done", encoding='ascii')
    elif command == "MarketState":
        resultStr = str(dynamicMarketState.curUpDowns)
        socket.send_string(resultStr, encoding='ascii')
