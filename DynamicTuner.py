import numpy as np
from sklearn.neural_network import MLPClassifier


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

    def Add(self, isBottom, resultStr ):
        print( "I will add: ", resultStr )
        resultStrList = resultStr.split(";")
        resultStrList2 = map(lambda x: x[2:-2].split(" ")[1], resultStrList[1:self.totalTransactionCount+1])
        transactions = list(map(float, resultStrList2))
        print( " Transactions ", transactions )
        self.inputResults.append(transactions)
        if isBottom :
            self.realResults.append(1)
            self.goodCount += 1
        else:
            self.realResults.append(0)
            self.badCount += 1

        curResultCount = self.goodCount + self.badCount

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
        if self.lastTrainNumber < 2 or self.goodCount < 8 or self.badCount < 8:
            return "[[1 -1]]"
        else:
            print("Will predict the fusion with request ", request)
            featureArr = np.array(request).reshape(-1, 1)
            return str(self.transactionTuneLearner.predict_proba(featureArr))

