import numpy as np
from sklearn.neural_network import MLPClassifier
import InputManager
import TransactionHelper as transHelper
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from scipy import stats
import sys

def MergeTransactions ( transactionList, msec, transactionBinCount, smallestTime ):
    index = msec // smallestTime
    totalElement = index * transactionBinCount
    arrayList = np.array_split(transactionList[-totalElement:], transactionBinCount)
    mergeArray = list(map(lambda x: x.sum(), arrayList))
    summedArray = list(map(lambda x: transHelper.NormalizeTransactionCount(x), mergeArray))
    return summedArray

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


def FindBestPoint( badData, goodData , gram ):
    bestVal = sys.maxsize
    bestIndex = 0
    bestElimanateGoodCount = 0

    for curBadIndex in range(np.size(badData)//2):
        curScore = badData[curBadIndex]
        eliminateGoodCount = np.count_nonzero(goodData < curScore)
        curVal = curBadIndex+eliminateGoodCount
        if curVal < bestVal:
            bestIndex = curBadIndex
            bestVal = curVal
            bestElimanateGoodCount = eliminateGoodCount

    print( " Gram: ", gram, " Best index: ", bestIndex, " Best score: ", badData[bestIndex], " Best val: ", bestVal, " Eliminate good: ", bestElimanateGoodCount)
    return badData[bestIndex]


def AdjustTheBestCurve( mergeResults, realResults, finalResult):
    goodIndexes = [i for i, x in enumerate(realResults) if x == 1.0 ]
    badIndexes = [i for i, x in enumerate(realResults) if x == 0.0 ]

    resultBadGoodAvarageDiff = []
    goodDataList = []
    badDataList = []

    print( " Current shape : ", mergeResults.shape, " Good count: ", len(goodIndexes), " Bad count: ", len(badIndexes) )
    if len(badIndexes) < 5:
        return
    if mergeResults.shape[1] == 0:
        return
    for col in range(mergeResults.shape[1]):
        curCol = mergeResults[:, col]
        goodData = -np.sort(-np.take(curCol, goodIndexes))
        badData = -np.sort(-np.take(curCol, badIndexes))
        goodDataTrimmed = stats.trim1(goodData, 0.05,  tail='left')
        badDataTrimmed = stats.trim_mean(badData, 0.05)

        goodAvarage = np.mean(goodDataTrimmed)
        badAvarage = np.mean(badDataTrimmed)
        resultBadGoodAvarageDiff.append(  goodAvarage - 2 * badAvarage)
        goodDataList.append(goodData)
        badDataList.append(badData)

    sortedIndexes = argsort(resultBadGoodAvarageDiff)
    print(sortedIndexes)
    for gramIndex in range(len(sortedIndexes)):
        sortedElem = sortedIndexes[-(gramIndex+1)]
        FindBestPoint(badDataList[sortedElem], goodDataList[sortedElem], sortedElem)

    bestIndex = sortedIndexes[-1]
    bestPoint = FindBestPoint(badDataList[bestIndex], goodDataList[bestIndex], bestIndex)

    bestIndexReal = bestIndex
    for i in range(len(finalResult)):
        if finalResult[i] != 0.0:
            bestIndexReal += 1
        if bestIndexReal == i:
            break

    print(" Best point: ", bestPoint, " Real: ", bestIndexReal, " Final Result: ", finalResult)
    finalResult[bestIndexReal] = bestPoint
    removeList = np.where(mergeResults[:, bestIndex] < bestPoint)
    mergeResultsNew = np.delete(mergeResults, removeList, 0)
    realResultsNew =  np.delete(realResults, removeList)
    print ( " Best elem was : ", bestIndex , " Now I will remove and try again ")
    mergeResultsNew = np.delete(mergeResultsNew, bestIndex , 1)
    AdjustTheBestCurve(mergeResultsNew, realResultsNew, finalResult)



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


    def GetCurrentResult(self):
        return str(self.inputResults) + "|" + str(self.realResults)

    def Init(self, inputShaper, mlpTransactionScalerList, mlpTransactionList, transList):
        transactionFeatures = inputShaper.toTransactionFeaturesNumpy(0)
        results = inputShaper.toTransactionResultsNumpy(0)
        resultPredicts = [[] for _ in range(len(transList))]
        for curIndex in range(len(transList)):
            trans = transList[curIndex]
            currentData = []

            for elem in transactionFeatures:
                justTransactions = elem[:-6]
                extras = elem[-6:]
                newSum = MergeTransactions(justTransactions, trans.msec, trans.gramCount, 250)
                currentData.extend(list(newSum)+list(extras))

            featureArr = np.array(currentData)
            featureArr = featureArr.reshape(-1, trans.gramCount+6)
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

        print("Tuner good size" , sum( y > 0 for y in self.realResults ),  " Total size ", len(self.realResults) )
        finalResult = [0.0] * totalResult.shape[1]
        AdjustTheBestCurve(totalResult, results, finalResult)
        print("Tuner final result is: ", finalResult)
        self.lastTrainNumber = len(self.realResults) // 30


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
        elemCount = curResultCount // 30
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

