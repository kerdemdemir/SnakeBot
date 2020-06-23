import json
import datetime
import bisect
import copy

class TransactionData:
    def __init__(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.normalizedCount = 0.0
        self.score = 0

    def __repr__(self):
        return "TotalBuy:%f,TotalSell:%f,TransactionCount:%f,Score:%f, NormalizedCount:%d" % (self.totalBuy, self.totalSell,
                                                                                            self.transactionCount,self.score,
                                                                                              self.normalizedCount)

    def NormalizeTransactionCount(self):
        if self.transactionCount < 2:
            self.normalizedCount = 0
            return
        elif self.transactionCount < 5:
            self.normalizedCount = 1
            return
        elif self.transactionCount < 10:
            self.normalizedCount = 2
            return
        elif self.transactionCount < 15:
            self.normalizedCount = 3
            return
        elif self.transactionCount < 25:
            self.normalizedCount = 4
            return
        elif self.transactionCount < 50:
            self.normalizedCount = 5
            return
        elif self.transactionCount < 100:
            self.normalizedCount = 6
            return
        elif self.transactionCount < 200:
            self.normalizedCount = 7
            return
        self.normalizedCount = 8

    #"m": true, "l": 6484065,"M": true,"q": "44113.00000000","a": 5378484,"T": 1591976004949,"p": "0.00000225","f": 6484064
    def AddData(self, jsonIn):
        isSell = jsonIn["m"]
        power  = float(jsonIn["q"]) * float(jsonIn["p"])
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
        self.maxNormalizedCount = 0

    def Append( self, dataList ):
        for elem in dataList:
            elem.NormalizeTransactionCount()
            self.transactionList.append(elem.normalizedCount)
            self.maxNormalizedCount = max( self.maxNormalizedCount, elem.normalizedCount)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionCount

    def GetFeatures(self):
        returnval = self.transactionList + [self.BuyVsSellRatio()]
        return returnval


    def BuyVsSellRatio(self):
        if self.totalSell == 0.0:
            return 4
        buySellRatio = self.totalBuy//self.totalSell
        return min(4,buySellRatio)

    def __repr__(self):
        return "list:%s,normalized:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f, maxNormalizedCount:%d" % (
                                                                                str(self.transactionList), self.BuyVsSellRatio(),
                                                                                self.totalBuy, self.totalSell,
                                                                                self.transactionCount,self.score,
                                                                                self.maxNormalizedCount )

    def __eq__(self, another):
        return len(self.transactionList) == len(another.transactionList) and \
               self.transactionList == another.transactionList and another.BuyVsSellRatio() == self.BuyVsSellRatio()


    def __hash__(self):
        resultHash = int(0)
        for i in range (0, len(self.transactionList)):
            resultHash += int(int(pow(10, i)) * int(self.transactionList[i]))
        return resultHash


class TransactionPeakHelper:
    percent = 0.01
    lowestAcceptedNormalizedTransactionCount = 2
    maxFeatureCount = 7
    minFeatureCount = 3

    def __init__(self, jsonIn, mseconds, isBottom ):
        self.mseconds = mseconds
        totalSize = len(jsonIn)
        self.patternList = [{} for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.dataList = []
        self.isBottom = isBottom

        prices = list(map( lambda x: float(x["p"]), jsonIn))
        if len(prices) == 0:
            return

        if isBottom:
            self.peakIndex = prices.index(min(prices))
        else:
            self.peakIndex = prices.index(max(prices))

        self.peakVal = float(jsonIn[self.peakIndex]["p"])
        self.startIndex = max(0, self.__FindIndexWithPriceAndPercent(jsonIn, self.peakIndex, -1, -1))
        self.stopIndex = min(totalSize, self.__FindIndexWithPriceAndPercent(jsonIn, self.peakIndex, totalSize,1))
        self.__DivideDataInSeconds(jsonIn)

    def GetTransactionPatterns(self, featureCount):
        transactionPatterns = self.patternList[featureCount].keys()
        returnVal = []
        for transactionPattern in transactionPatterns:
            returnVal.append( transactionPattern.GetFeatures() )
        return returnVal

    #TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
    def AssignScores( self ):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        print(lenArray, self.dataList)
        for x in range( 0, lenArray ):
            curElement = self.dataList[x]
            curElement.NormalizeTransactionCount()
            for curBin in range(self.maxFeatureCount - self.minFeatureCount):
                self.__AppendToPatternList(curBin, x, lenArray)

    def __FindIndexWithPriceAndPercent(self, jsonIn, curStartIndex, curStopIndex, step):
        for x in range(curStartIndex, curStopIndex, step ):
            curElem = jsonIn[x]
            price = float(curElem["p"])
            if self.isBottom:
                if price > self.peakVal * (1.00+self.percent):
                    return x
            else:
                if price < self.peakVal * (1.00-self.percent):
                    return x
        return 0 if step == -1 else len(jsonIn)

    def __DivideDataInSeconds(self, jsonIn):
        transactionData = TransactionData()
        lastTime = 0
        for x in range(self.startIndex, self.stopIndex):
            curElement = jsonIn[x]
            curMiliSecs = int(curElement["T"])
            if x == self.startIndex:
                lastTime = curMiliSecs

            diffTime = curMiliSecs - lastTime
            if diffTime > self.mseconds:
                lastTime = curMiliSecs
                self.dataList.append( copy.deepcopy(transactionData) )
                transactionData.Reset()
                transactionData.AddData(curElement)
            else:
                transactionData.AddData(curElement)
        self.dataList.append(copy.deepcopy(transactionData))

    def __AppendToPatternList(self, curBin, curIndex, lenArray):
        ngramCount = curBin + self.minFeatureCount
        for index in range(ngramCount):
            startBin = curIndex + 1 - ngramCount + index
            endBin = curIndex + 1 + index
            if startBin < 0 or endBin > lenArray:
                continue

            pattern = TransactionPattern()
            pattern.Append(self.dataList[startBin:endBin])

            if pattern.maxNormalizedCount < TransactionPeakHelper.lowestAcceptedNormalizedTransactionCount:
                continue

            if pattern in self.patternList[curBin]:
                #print("Adding a existing element: ", self.dataList[startBin:endBin])
                self.patternList[curBin][pattern] += 1 if self.isBottom else -1
            else:
                #print("Adding a new element: ", self.dataList[startBin:endBin])
                self.patternList[curBin][pattern] = 1 if self.isBottom else -1



class TransactionAnalyzer :
    def __init__(self ):
        self.patternList = [{} for _ in range(TransactionPeakHelper.maxFeatureCount - TransactionPeakHelper.minFeatureCount + 1)]
        self.peakFeatures = [] # 3D array for each peak there is a  array of 4 grams

    def AddPeak(self, jsonIn, riseMinuteList, msec, ngrams ):
        self.peakFeatures.clear()
        for index in range(len(jsonIn)):
            isBottom = riseMinuteList[index].rise < 0.0
            peakHelper = TransactionPeakHelper(jsonIn[index], msec, isBottom)
            peakHelper.AssignScores()
            self.__MergeInTransactions(peakHelper, isBottom)
            gramIndex = ngrams - TransactionPeakHelper.minFeatureCount
            self.peakFeatures.append( peakHelper.GetTransactionPatterns(gramIndex) )

    def Print( self ):
        for index in range(TransactionPeakHelper.maxFeatureCount - TransactionPeakHelper.minFeatureCount):
            ngramCount = index + TransactionPeakHelper.minFeatureCount
            #peakPatternList = self.patternList[timeIndex][index].keys()
            #for pattern in peakPatternList:
            #    print("Seconds", timeIndex, " " , ngramCount, " ", pattern, " val: ",  self.patternList[timeIndex][index][pattern])
            peakPatternValues = list(self.patternList[index].values())
            peakPatternValues.sort()
            m = sum(peakPatternValues) / len(peakPatternValues)
            var_res = sum((xi - m) ** 2 for xi in peakPatternValues) / len(peakPatternValues)
            print(" gramCount: ", ngramCount, " len: ", len(peakPatternValues), " small: ", peakPatternValues[0],
                  " last: ", peakPatternValues[-1], " mean ", m, " var ", var_res)



    def __MergeInTransactions(self, transactionPeakHelper, isBottom ):
        for index in range(TransactionPeakHelper.maxFeatureCount-TransactionPeakHelper.minFeatureCount):
            # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
            peakPatternList = transactionPeakHelper.patternList[index].keys()
            for pattern in peakPatternList:
                if pattern in self.patternList[index]:
                    print("Merging a existing element: ", pattern)
                    self.patternList[index][pattern] += (1 if isBottom else -1)
                else:
                    print("Merging a new element: ", pattern)
                    self.patternList[index][pattern] = (1 if isBottom else -1)

