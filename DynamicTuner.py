import numpy as np
from sklearn.neural_network import MLPClassifier
import InputManager
import TransactionHelper as transHelper

def MergeTransactions ( transactionList, msec, transactionBinCount, smallestTime ):
    index = msec // smallestTime
    totalElement = index * transactionBinCount
    arrayList = np.array_split(transactionList[-totalElement:], transactionBinCount)
    mergeArray = list(map(lambda x: x.sum(), arrayList))
    summedArray = list(map(lambda x: transHelper.NormalizeTransactionCount(x), mergeArray))
    return summedArray


class PeakTransactionTurner:
    def __init__(self, totalTransactionCount):
        self.inputResults = []
        self.realResults = []
        self.totalTransactionCount = totalTransactionCount
        self.lastTrainNumber = 0
        self.goodCount = 0
        self.badCount = 0
        self.transactionTuneLearner = MLPClassifier(hidden_layer_sizes=(5, 5, 5), activation='relu',
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
                newSum = MergeTransactions(justTransactions, trans.msec, trans.gramCount, 125)
                currentData.extend(list(newSum)+list(extras))

            featureArr = np.array(currentData)
            featureArr = featureArr.reshape(-1, trans.gramCount+6)
            X = mlpTransactionScalerList[curIndex].transform(featureArr)
            curResultPredict = mlpTransactionList[curIndex].predict_proba(X)
            resultPredicts[curIndex] = np.delete(curResultPredict, 0, 1)


        totalResult = np.empty([np.size(resultPredicts,1),np.size(resultPredicts,0)])
        for curIndex in range(len(transList)):
            np.append(totalResult, resultPredicts[curIndex], axis=1)

        for elem in totalResult:
            self.inputResults.append(elem)

        self.realResults = list(results)
        self.transactionTuneLearner.fit(totalResult, results)
        print("Tuner good size" , sum( y > 0 for y in self.realResults ),  " Total size ", len(self.realResults) )
        self.lastTrainNumber = len(self.realResults) // 10


    def Add(self, isBottom, resultStr ):
        print( "I will add: ", resultStr )
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
        elemCount = curResultCount // 10
        if elemCount == self.lastTrainNumber:
            return
        self.lastTrainNumber = elemCount
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

