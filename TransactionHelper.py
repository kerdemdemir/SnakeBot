import json
import time
import bisect


class TransactionData:
    def __init__(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0

    def NormalizeTransactionCount(self):
        if self.transactionCount < 5:
            self.transactionCount = 0
        elif self.transactionCount < 10:
            self.transactionCount = 1
        elif self.transactionCount < 15:
            self.transactionCount = 2
        elif self.transactionCount < 25:
            self.transactionCount = 3
        elif self.transactionCount < 50:
            self.transactionCount = 4
        elif self.transactionCount < 100:
            self.transactionCount = 5
        elif self.transactionCount < 200:
            self.transactionCount = 6
        elif self.transactionCount < 500:
            self.transactionCount = 7
        self.transactionCount = 8

    #"m": true, "l": 6484065,"M": true,"q": "44113.00000000","a": 5378484,"T": 1591976004949,"p": "0.00000225","f": 6484064
    def AddData(self, jsonIn):
        isSell = jsonIn["m"]
        power  = float(jsonIn["m"]) * float(jsonIn["p"])
        if not isSell:
            self.transactionCount += 1
            self.totalBuy += power
        else:
            self.totalSell += power

    def Reset(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0

class TransactionPattern:
    def __init__(self):
        self.transactionList = []
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0.0

    def Append( self, dataList ):
        for elem in dataList:
            self.transactionList.append(elem.transactionCount)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionCount


class TransactionPeakHelper:
    percent = 0.01
    lowestAcceptedNormalizedTransactionCount = 1
    maxFeatureCount = 7
    minFeatureCount = 3

    def __init__(self, jsonIn, mseconds, isBottom, peakTime ):
        self.mseconds = mseconds
        self.peakIndex =  self.peakIndex = bisect.bisect( jsonIn, peakTime)
        self.isBottom = isBottom
        self.startIndex = max(0, self.__FindIndexWithPriceAndPercent(self.peakIndex, 0))
        totalSize = len(jsonIn)
        self.stopIndex = min(totalSize, self.__FindIndexWithPriceAndPercent(self.peakIndex, totalSize))
        self.peakVal = float(json[self.peakIndex])
        self.dataList = []
        self.patternList = [{} for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.__DivideDataInSeconds(jsonIn)


    #TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
    def AssignScores( self ):
        lenArray = len(self.dataList)
        for x in ( 0, lenArray ):
            curElement = self.dataList[x]
            curElement.NormalizeTransactionCount()
            if curElement.transactionCount >= TransactionPeakHelper.lowestAcceptedNormalizedTransactionCount:
                for curBin in range(self.maxFeatureCount - self.minFeatureCount):
                    self.__AppendToPatternList(curBin, x, lenArray)

    def __FindIndexWithPriceAndPercent(self, jsonIn, curStartIndex, curStopIndex):
        for x in range(curStartIndex, curStopIndex ):
            price = float(jsonIn[x]["p"])
            if self.isBottom:
                if price > self.peakVal * (1.00+self.percent):
                    return x
            else:
                if price < self.peakVal * (1.00-self.percent):
                    return x
        return self.peakIndex

    def __DivideDataInSeconds(self, jsonIn):
        lastTime = time.time()
        transactionData = TransactionData();
        for x in range(self.startIndex, self.stopIndex):
            curElement = jsonIn[x]
            curTime = time.ctime(curElement["T"])
            if x == self.startIndex:
                lastTime = curTime
            if curTime - lastTime > self.mseconds:
                lastTime = curTime
                self.dataList.append(transactionData)
                transactionData.reset()
            else:
                transactionData.AddData(curElement)

    def __AppendToPatternList(self, curBin, curIndex, lenArray):
        ngramCount = curBin + self.minFeatureCount
        for index in range(ngramCount):
            startBin = curIndex - ngramCount + index
            endBin = curIndex + index
            if startBin < 0 or endBin > lenArray:
                continue
            pattern = TransactionPattern()
            pattern.Append(self.dataList[startBin:endBin])
            if pattern in self.patternList[curBin]:
                self.patternList[curBin][pattern] += 1 if self.isBottom else -1
            else:
                self.patternList[curBin][pattern] = 1 if self.isBottom else -1



class TransactionAnalyzer :
    def __init__(self ):
        self.msecondList = [ 3000, 6000, 10000 ]
        self.patternList = [{} for _ in range(self.maxFeatureCount - self.minFeatureCount)]

    def AddPeak(self, jsonIn, riseMinuteList ):
        for msec in self.msecondList:
            for index in range(len(jsonIn)):
                peakHelper = TransactionPeakHelper(jsonIn[index], msec, riseMinuteList[index].rise < 0.0, riseMinuteList[index].time)
                self.__MergeInTransactions(peakHelper)

    def __MergeInTransactions(self, transactionPeakHelper):
        for index in range(TransactionPeakHelper.maxFeatureCount):
            curNGram = index + TransactionPeakHelper.minFeatureCount

            # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
            peakPatternList = transactionPeakHelper.patternList[index].keys()
            for pattern in peakPatternList:
                if pattern in self.patternList[index]:
                    self.patternList[index][pattern] += (1 if self.isBottom else -1)
                else:
                    self.patternList[index][pattern] = (1 if self.isBottom else -1)

