import ExtraDataManager as extraDataMan
import MarketStateManager as marketState
import BuySellAnalyzer as buyAnalyzer

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
transParamList = [TransactionBasics.TransactionParam(1000, 12)]

currentProbs = []

def TrainAnaylzer():
    falsePositives = []
    truePositives = []
    for i in range(25):
        mlpTransactionList.clear()
        mlpTransactionScalerList.clear()
        curResult = Learn()
        falsePositives.append(int(curResult[0][1]))
        truePositives.append(int(curResult[1][1]))

        print(falsePositives, " ", truePositives)

    badListArray = np.array(falsePositives)
    goodListArray = np.array(truePositives)
    badLegend = str(np.quantile(badListArray, 0.1)) + ", " + str(np.quantile(badListArray, 0.25)) + " , ** " \
                + str(np.quantile(badListArray, 0.5)) + " ** ," + str(np.quantile(badListArray, 0.75)) + " , " + str(
        np.quantile(badListArray, 0.9))
    goodLegend = str(np.quantile(goodListArray, 0.1)) + " , " + str(np.quantile(goodListArray, 0.25)) + \
                 " , ** " + str(np.quantile(goodListArray, 0.5)) + " ** , " + str(np.quantile(goodListArray, 0.75)) + " , " + str(
        np.quantile(goodListArray, 0.9))
    print(" Good results ", goodLegend)
    print(" Bad results ", badLegend)

def Predict ( messageChangeTimeTransactionStrList, mlpTransactionScalerListIn, mlpTransactionListIn, isBuySell, isAvoidPeaks ):

    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")
    extrasStrList = messageChangeTimeTransactionStrList[3].split(",")
    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsExtraFloat = [float(extraStr) for extraStr in extrasStrList]

    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    extraMaxMinList = []
    if not isBuySell:
        marketStateList = dynamicMarketState.curUpDowns
        if not isAvoidPeaks:
            extraMaxMinList = TransactionBasics.GetMaxMinList(resultsExtraFloat)
    else:
        marketStateList = dynamicMarketState.getNowAndBuyState(resultsExtraFloat[1])
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
        if TransactionBasics.PeakFeatureCount == 0 or isBuySell :
            totalFeatures = currentTransactionList + marketStateList #+ resultsExtraFloat
        else:
            totalFeatures = currentTransactionList + marketStateList +\
                            resultsChangeFloat[-TransactionBasics.PeakFeatureCount:] + \
                            resultsTimeFloat[-TransactionBasics.PeakFeatureCount:] + extraMaxMinList


        totalFeaturesNumpy = np.array(totalFeatures).reshape(1, -1)
        totalFeaturesScaled = mlpTransactionScalerListIn[transactionIndex].transform(totalFeaturesNumpy)
        print("I will predict: ", totalFeatures, " scaled: ", totalFeaturesScaled)
        npTotalFeatures = np.array(totalFeaturesScaled)
        npTotalFeatures = npTotalFeatures.reshape(1, -1)
        predict_test = mlpTransactionListIn[transactionIndex].predict_proba(npTotalFeatures)
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


        mlpTransaction = MLPClassifier(hidden_layer_sizes=(36, 36, 36, 36, 36), activation='relu',
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
        returnResult = confusion_matrix(y_test, finalResult)
        print("60 ", returnResult)
        #print(predict_test)
        #predict_test = np.delete(finalResult, 0, 1)

        print(" Transactions time: ", transParam.msec, " Transaction Index ", transParam.gramCount, "Index ",
              transactionIndex)

        sys.stdout.flush()
        return returnResult


def LearnWhenToSell():
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        numpyArr = buySellDataManager.toSellTransactions(transactionIndex)
        mlpTransaction = MLPClassifier(hidden_layer_sizes=(24, 24, 24, 24), activation='relu',
                                       solver='adam', max_iter=500)
        mlpTransactionListSell.append(mlpTransaction)
        transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
        mlpTransactionScalerListSell.append(transactionScaler)
        X = transactionScaler.transform(numpyArr)
        y = buySellDataManager.toSellResultsNumpy(transactionIndex)  # + extraDataManager.getResult(transactionIndex)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

        mlpTransaction.fit(X_train, y_train)

        predict_test = mlpTransaction.predict_proba(X_test)
        finalResult = predict_test[:, 1] >= 0.6
        returnResult = confusion_matrix(y_test, finalResult)
        print("60 ", returnResult)

        print(" Sell Transactions time: ", transParam.msec, " Sell Transaction Index ", transParam.gramCount, " Sell Index ",
              transactionIndex)

        sys.stdout.flush()
        return returnResult


def LearnAvoidPeak():
    for transactionIndex in range(len(transParamList)):
        transParam = transParamList[transactionIndex]
        numpyArr = suddenChangeManager.toSellTransactions(transactionIndex)
        mlpTransaction = MLPClassifier(hidden_layer_sizes=(24, 24, 24, 24), activation='relu',
                                       solver='adam', max_iter=500)
        mlpTransactionListAvoidPeak.append(mlpTransaction)
        transactionScaler = preprocessing.StandardScaler().fit(numpyArr)
        mlpTransactionScalerListAvoidPeak.append(transactionScaler)
        X = transactionScaler.transform(numpyArr)
        y = suddenChangeManager.toSellResultsNumpy(transactionIndex)  # + extraDataManager.getResult(transactionIndex)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)

        mlpTransaction.fit(X_train, y_train)

        predict_test = mlpTransaction.predict_proba(X_test)
        finalResult = predict_test[:, 1] >= 0.6
        returnResult = confusion_matrix(y_test, finalResult)
        print("60 ", returnResult)

        print(" Sell 2 Transactions time: ", transParam.msec, " Sell 2 Transaction Index ", transParam.gramCount, " Sell 2 Index ",
              transactionIndex)

        sys.stdout.flush()
        return returnResult


dynamicMarketState = marketState.MarketStateManager()
suddenChangeManager = SuddenChangeTransactions.SuddenChangeManager(transParamList)
if isUsePeaks:
    peakManager = PeakTransactions.PeakManager(transParamList)
if isUseExtraData:
    extraFolderPath = os.path.abspath(os.getcwd()) + "/Data/ExtraData/"
    extraDataManager = extraDataMan.ExtraDataManager(extraFolderPath,transParamList,suddenChangeManager.marketState)

mlpTransactionList = []
mlpTransactionScalerList = []
#TrainAnaylzer()
curResult = Learn()

buySellDataManager = buyAnalyzer.BuyAnalyzeManager(transParamList, suddenChangeManager.marketState)
print("BuySellAnalyzeFinished")
sys.stdout.flush()

mlpTransactionListSell = []
mlpTransactionScalerListSell = []
LearnWhenToSell()

mlpTransactionListAvoidPeak = []
mlpTransactionScalerListAvoidPeak = []
LearnAvoidPeak()

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
        resultStr = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerList, mlpTransactionList, False, False)
        print("Results are: ", resultStr)
        #  Send reply back to client
        socket.send_string(resultStr, encoding='ascii')
        sys.stdout.flush()
    elif command == "ShouldSell":
        messageChangeTimeTransactionStrList = messageChangeTimeTransactionStrList[1:]
        resultStr = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerListSell, mlpTransactionListSell, True, False)
        print("Results are: ", resultStr)
        resultStr2 = Predict(messageChangeTimeTransactionStrList, mlpTransactionScalerListAvoidPeak, mlpTransactionListAvoidPeak, False, True)
        #  Send reply back to client
        totalResult = resultStr + "," + resultStr2
        socket.send_string(totalResult, encoding='ascii')
        sys.stdout.flush()
    elif command == "Peak":
        print( " New peak will be added ")
        isRising = messageChangeTimeTransactionStrList[1] == "Increase"
        dynamicMarketState.addRecent(isRising)
        socket.send_string("Done", encoding='ascii')
    elif command == "MarketState":
        resultStr = str(dynamicMarketState.curUpDowns)
        socket.send_string(resultStr, encoding='ascii')
