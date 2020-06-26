import numpy as np
import TransactionHelper
import itertools

class RiseMinute:
    def __init__(self, riseAndTimeStr):
        riseAndTimeStrPair = riseAndTimeStr.split("|")
        self.rise = float(riseAndTimeStrPair[0])
        self.time = riseAndTimeStrPair[1]

    def __repr__(self):
        return "Rise:%f,Time:%s" % (self.rise, self.time)


class ReShapedInput:

    def __init__(self, featureCount ):
        self.inputRise = []
        self.inputTime = []
        self.featureCount = featureCount
        self.featureArr = []
        self.transactionFeatures = []
        self.output = []

    def concanate(self, riseAndTimeList,lastTransactionFeatures):
        riseList = list(map( lambda x: float(x.rise), riseAndTimeList ))
        newList = list(map(lambda x: riseList[x:x+self.featureCount],
                                  range( len( riseList ) - self.featureCount)))

        timeList = list(map( lambda x: float(x.time), riseAndTimeList ))
        newTimeList = list(map(lambda x: timeList[x:x+self.featureCount],
                                  range( len( timeList ) - self.featureCount)))


        transactionFeatures = list(map(lambda x: itertools.chain(*lastTransactionFeatures[x:x+self.featureCount]),
                                  range( len( lastTransactionFeatures ) - self.featureCount)))

        self.inputRise.extend(newList)
        self.inputTime.extend(newTimeList)
        self.transactionFeatures.extend(transactionFeatures)

    def toNumpy(self):
        inputSize = len(self.inputRise)
        if np.size(self.featureArr, 0) == inputSize:
            return self.featureArr
        temp = []
        for curIndex in range(inputSize):
            newRow = self.inputRise[curIndex] + self.inputTime[curIndex]
            temp.append(newRow)
            outputVal = 1 if 0.0 < self.inputRise[curIndex][-1] else 0
            self.output.append(outputVal)
        self.featureArr = np.array(temp)
        self.featureArr.reshape(-1, self.featureCount*2)
        return self.featureArr

    def toTransactionNumpy(self, transactionCount):
        inputSize = len(self.inputRise)
        temp = []
        for curIndex in range(inputSize):
            for transactions in self.transactionFeatures[curIndex]:
                newRow = self.inputRise[curIndex][2 - self.featureCount:] + self.inputTime[curIndex][2 - self.featureCount:] + transactions
                temp.append(newRow)
                #print(newRow)
                outputVal = 1 if 0.0 < self.inputRise[curIndex][-1] else 0
                self.output.append(outputVal)
        self.featureArr = np.array(temp)
        self.featureArr.reshape(-1, (self.featureCount-2)*2+transactionCount+1)
        return self.featureArr