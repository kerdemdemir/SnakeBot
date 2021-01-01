import numpy as np
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy import stats
import sys

smallestTime = 500
totalTransactions = 30
startTime = 0
listCount = 4

def MergeTransactions ( transactionList, msec, transactionBinCount ):
    index = (msec - startTime) // smallestTime
    totalElement = index * transactionBinCount * listCount
    if totalElement >= len(transactionList):
        return []
    arrayList = transactionList[-totalElement:]
    mergeArray = []
    for i in range(transactionBinCount):
        curStartIndex = i * listCount * index
        for j in range(listCount):
            mergeArray.append(0)
            for k in range(index):
                mergeArray[-1] += arrayList[curStartIndex + j + k * listCount]

    return mergeArray


def PredictionPrice( acceptanceLevel, predict_test, y_test ):
    finalResult = predict_test[:, 1] >= acceptanceLevel
    print(acceptanceLevel, " ", confusion_matrix(y_test, finalResult))

def FitPredictAndPrint( leaner, X_train, X_test, y_train, y_test):
    leaner.fit(X_train, y_train)
    predict_test = leaner.predict_proba(X_test)
    PredictionPrice(0.5, predict_test, y_test)
    PredictionPrice(0.6, predict_test, y_test)
    PredictionPrice(0.7, predict_test, y_test)
    PredictionPrice(0.8, predict_test, y_test)
    PredictionPrice(0.9, predict_test, y_test)

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def FindBestPoint( badData, goodData , gram, factor = 1 ):
    bestVal = sys.maxsize
    bestIndex = 0
    bestElimanateGoodCount = 0
    badSize = np.size(badData)
    for curBadIndex in range(badSize):
        curScore = badData[curBadIndex]
        eliminateGoodCount = np.count_nonzero(goodData < curScore)
        #leftOverBadCount = badSize - curBadIndex
        curVal = curBadIndex+(eliminateGoodCount*factor)
        if curVal < bestVal:
            bestIndex = curBadIndex
            bestVal = curVal
            bestElimanateGoodCount = eliminateGoodCount

    print( " Gram: ", gram, " Best index: ", bestIndex, " Best score: ", badData[bestIndex], " Best val: ", bestVal, " Eliminate good: ", bestElimanateGoodCount)
    return badData[bestIndex]


def AdjustTheBestCurve( mergeResults, realResults, finalResult, startIndex = -1):
    goodIndexes = [i for i, x in enumerate(realResults) if x == 1.0 ]
    badIndexes = [i for i, x in enumerate(realResults) if x == 0.0 ]

    resultBadGoodAvarageDiff = []
    goodDataList = []
    badDataList = []

    firstGoodCount = len(goodIndexes)
    firstBadCount = len(badIndexes)

    print( " Current shape : ", mergeResults.shape, " Good count: ", firstGoodCount, " Bad count: ", firstBadCount )
    if len(badIndexes) < 5:
        return
    if mergeResults.shape[1] == 0:
        return
    for col in range(mergeResults.shape[1]):
        curCol = mergeResults[:, col]
        goodData = -np.sort(-np.take(curCol, goodIndexes))
        badData = -np.sort(-np.take(curCol, badIndexes))


        goodAvarage = np.mean(goodData)
        badAvarage = np.mean(badData)
        resultBadGoodAvarageDiff.append(  goodAvarage -  badAvarage)
        goodDataList.append(goodData)
        badDataList.append(badData)


    sortedIndexes = argsort(resultBadGoodAvarageDiff)

    print(sortedIndexes)

    bestIndex = sortedIndexes[-1]
    if startIndex != -1:
        print("Best index was: ", bestIndex, " But that index forced instead: ", startIndex)
        bestIndex = startIndex
    eliminateFactor = 1.5 if startIndex != -1 else 1.2
    bestPoint = FindBestPoint(badDataList[bestIndex], goodDataList[bestIndex], bestIndex, eliminateFactor)
    bestIndexReal = bestIndex
    for i in range(len(finalResult)):
        if finalResult[i] != 0.0:
            bestIndexReal += 1
        if bestIndexReal == i:
            break

    if bestPoint > 0.85:
        print("Results was really big I am normalizing to 0.85 index:", bestIndexReal )
        bestPoint = 0.85

    finalResult[bestIndexReal] = bestPoint
    removeList = np.where(mergeResults[:, bestIndex] < bestPoint)
    mergeResultsNew = np.delete(mergeResults, removeList, 0)
    realResultsNew =  np.delete(realResults, removeList)

    curGoodCount =  len([i for i, x in enumerate(realResultsNew) if x == 1.0 ])
    curBadCount  =  len([i for i, x in enumerate(realResultsNew) if x == 0.0 ])
    if firstGoodCount - curGoodCount > firstBadCount - curBadCount :
            finalResult[bestIndexReal] = 0.001
            bestPoint = 0.001
            print( " There are more  good result removed than bad results removed goods: ", firstGoodCount - curGoodCount,
                   " bads: ", firstBadCount - curBadCount, " Real index, ", bestIndexReal)
    elif (firstBadCount - curBadCount) < 10:
        finalResult[bestIndexReal] = 0.001
        bestPoint = 0.001
        print( " Too little bad eliminated goods: ", firstGoodCount - curGoodCount, " bads: ", firstBadCount - curBadCount,
               " Real index, ", bestIndexReal )
    else:
        print( "Index ", bestIndexReal, "eliminated bads: ", firstBadCount - curBadCount, " eliminated goods: ",
               firstGoodCount - curGoodCount)


    print(" Best point: ", bestPoint, " Real: ", bestIndexReal, " Final Result: ", finalResult)
    mergeResultsNew = np.delete(mergeResultsNew, bestIndex , 1)
    AdjustTheBestCurve(mergeResultsNew, realResultsNew, finalResult, -1)

def ForceTheBestCurve( mergeResults, realResults, finalResultForced):
    mergeResultsNew = mergeResults
    realResultsNew = realResults
    indexes = np.where((mergeResultsNew < finalResultForced).any(axis=1))
    resultTemp = np.delete(realResultsNew, indexes, 0)

    curGoodCount = len([i for i, x in enumerate(resultTemp) if x == 1.0])
    curBadCount = len([i for i, x in enumerate(resultTemp) if x == 0.0])
    print(" Second Forcing method: ", curGoodCount, " ", curBadCount)


class PeakTransactionTurner:
    def __init__(self, totalTransactionCount):
        self.inputResults = []
        self.realResults = []
        self.totalTransactionCount = totalTransactionCount
        self.lastTrainNumber = 0
        self.goodCount = 0
        self.badCount = 0
        self.transactionTuneLearner = MLPClassifier(hidden_layer_sizes=(6,6,6), activation='relu',
                                                solver='adam', max_iter=500)
        self.finalResult = []


    def GetCurrentResult(self):
        return str(self.inputResults) + "|" + str(self.realResults)

    def Init(self, inputShaper, mlpTransactionScalerList, mlpTransactionList, transList):
        transactionFeatures = inputShaper.toTransactionFeaturesNumpy(0)
        results = inputShaper.toTransactionResultsNumpy(0)
        resultPredicts = [[] for _ in range(len(transList))]
        for curIndex in range(len(transList)):
            trans = transList[curIndex]
            currentData = []
            totalExtraFeatures = transHelper.ExtraFeatureCount + transHelper.TransactionPeakHelper.PeakFeatureCount

            for elem in transactionFeatures:
                justTransactions = elem[:-totalExtraFeatures]
                extras = elem[-totalExtraFeatures:]
                newSum = MergeTransactions(justTransactions, trans.msec, trans.gramCount)
                currentData.extend(list(newSum)+list(extras))

            featureArr = np.array(currentData)
            featureArr = featureArr.reshape(-1, trans.gramCount+totalExtraFeatures)
            if featureArr.size == 0:
                continue
            X = mlpTransactionScalerList[curIndex].transform(featureArr)
            curResultPredict = mlpTransactionList[curIndex].predict_proba(X)

            resultPredicts[curIndex] = np.delete(curResultPredict, 0, 1)

        totalResult = resultPredicts[0]
        for curIndex in range(len(transList)-1):
           totalResult = np.concatenate((totalResult, resultPredicts[curIndex+1]), axis=1)

        for elem in totalResult:
            self.inputResults.append(elem)

        self.realResults = list(results)

        X_train, X_test, y_train, y_test = train_test_split(totalResult, results, test_size=0.1, random_state=40)

        FitPredictAndPrint(self.transactionTuneLearner, X_train, X_test, y_train, y_test)
        self.lastTrainNumber = len(self.realResults) // 60
        print("Tuner good size" , sum( y > 0 for y in self.realResults ),  " Total size ", len(self.realResults) )

        forceList = [0.5, 0.5, 0.9, 0.5, 0.5, 0.9]
        ForceTheBestCurve(totalResult, results, forceList)

    def InitWithScore(self, inputShaper, mlpTransactionScalerList, mlpTransactionList, transList, score ):
        transactionFeatures = inputShaper.toTransactionFeaturesWithScoreNumpy(0, score)
        results = inputShaper.toTransactionResultsNumpy(0)
        resultPredicts = [[] for _ in range(len(transList))]
        for curIndex in range(len(transList)):
            trans = transList[curIndex]
            currentData = []
            totalExtraFeatures = transHelper.ExtraFeatureCount + transHelper.TransactionPeakHelper.PeakFeatureCount

            for elem in transactionFeatures:
                justTransactions = elem[:-totalExtraFeatures]
                extras = elem[-totalExtraFeatures:]
                newSum = MergeTransactions(justTransactions, trans.msec, trans.gramCount)
                currentData.extend(list(newSum) + list(extras))

            featureArr = np.array(currentData)
            featureArr = featureArr.reshape(-1, trans.gramCount + totalExtraFeatures)
            if featureArr.size == 0:
                continue
            X = mlpTransactionScalerList[curIndex].transform(featureArr)
            curResultPredict = mlpTransactionList[curIndex].predict_proba(X)

            resultPredicts[curIndex] = np.delete(curResultPredict, 0, 1)

        totalResult = resultPredicts[0]
        for curIndex in range(len(transList) - 1):
            totalResult = np.concatenate((totalResult, resultPredicts[curIndex + 1]), axis=1)

        self.inputResults.clear()
        for elem in totalResult:
            self.inputResults.append(elem)

        self.realResults = list(results)

        X_train, X_test, y_train, y_test = train_test_split(totalResult, results, test_size=0.1, random_state=40)

        FitPredictAndPrint(self.transactionTuneLearner, X_train, X_test, y_train, y_test)
        self.lastTrainNumber = len(self.realResults) // 60
        print("Tuner good size", sum(y > 0 for y in self.realResults), " Total size ", len(self.realResults))

        forceList = [0.5, 0.5, 0.9, 0.5, 0.5, 0.9]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.5, 0.5, 0.5, 0.5, 0.9]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.5, 0.7, 0.5, 0.5, 0.7]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.7, 0.5, 0.5, 0.5, 0.5]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.5, 0.5, 0.7, 0.5, 0.5]
        ForceTheBestCurve(totalResult, results, forceList)

        forceList = [0.5, 0.5, 0.5, 0.5, 0.7, 0.5]
        ForceTheBestCurve(totalResult, results, forceList)

    def Add(self, isBottom, resultStr ):
        print( "I will add isBottom: ", isBottom, " data: " , resultStr )
        resultStrList = resultStr.split(";")
        resultStrList2 = map(lambda x: x[2:-2].split(" ")[1], resultStrList[1:self.totalTransactionCount+1])
        transactions = list(map(float, resultStrList2))
        print( " Transactions ", transactions )
        self.inputResults.append(transactions)
        if isBottom :
            self.realResults.append(1)
        else:
            self.realResults.append(0)

        curResultCount = len(self.realResults)

        print("After addition cur results: ", curResultCount)
        elemCount = curResultCount // 60
        if elemCount == self.lastTrainNumber:
            return False
        return True


    def Train(self):
        print("Retraining: ")
        featureArr = np.array(self.inputResults)
        featureArr.reshape(-1, self.totalTransactionCount)

        resultArr = np.array(self.realResults)
        resultArr.reshape(-1, 1)

        self.transactionTuneLearner.fit(featureArr, resultArr)

    def GetResult( self, request ):
        if self.lastTrainNumber < 1:
            return "[[1 -1]]"
        else:
            print("Will predict the fusion with request ", request)

            featureArr = np.array(request).reshape(1, -1)
            return str(self.transactionTuneLearner.predict_proba(featureArr))

