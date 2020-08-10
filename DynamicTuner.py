import numpy as np
from sklearn.neural_network import MLPClassifier
import InputManager
import TransactionHelper as transHelper
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

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

        totalResult = resultPredicts[0]
        for curIndex in range(len(transList)-1):
            totalResult = np.concatenate((totalResult, resultPredicts[curIndex+1]), axis=1)

        for elem in totalResult:
            self.inputResults.append(elem)

        self.realResults = list(results)

        X_train, X_test, y_train, y_test = train_test_split(totalResult, results, test_size=0.1, random_state=40)

        self.transactionTuneLearner.fit(X_train, y_train)

        predict_test = self.transactionTuneLearner.predict_proba(X_test)
        finalResult = predict_test[:, 1] >= 0.5
        print("50 ",confusion_matrix(y_test, finalResult))

        finalResult = predict_test[:, 1] >= 0.6
        print("60 ", confusion_matrix(y_test, finalResult))

        finalResult = predict_test[:, 1] >= 0.7
        print("70 ", confusion_matrix(y_test, finalResult))

        finalResult = predict_test[:, 1] >= 0.8
        print("80 ", confusion_matrix(y_test, finalResult))

        finalResult = predict_test[:, 1] >= 0.9
        print("90 ", confusion_matrix(y_test, finalResult))


        self.transactionTuneLearner.fit(totalResult, results)
        print("Tuner good size" , sum( y > 0 for y in self.realResults ),  " Total size ", len(self.realResults) )
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

