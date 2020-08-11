import json
import datetime
import bisect
import copy
import numpy as np


def NormalizeTransactionCount( totalCount ):
    if totalCount < 1:
        return 0
    elif totalCount < 3:
        return 1
    elif totalCount < 5:
        return 2
    elif totalCount < 7:
        return 3
    elif totalCount < 10:
        return 4
    elif totalCount < 14:
        return 5
    elif totalCount < 19:
        return 6
    elif totalCount < 25:
        return 7
    elif totalCount < 50:
        return 8
    elif totalCount < 100:
        return 9
    elif totalCount < 200:
        return 10
    return 11


class TransactionData:
    def __init__(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.totalTransactionCount = 0.0
        self.normalizedCount = 0.0
        self.score = 0
        self.timeInSecs = 0
        self.firstPrice = 0.0
        self.lastPrice = 0.0

    def __repr__(self):
        return "TotalBuy:%f,TotalSell:%f,TransactionCount:%f,Score:%f, NormalizedCount:%d" % (self.totalBuy, self.totalSell,
                                                                                            self.transactionCount,self.score,
                                                                                              self.normalizedCount)


    def NormalizeTransactionCount(self):
        self.normalizedCount = NormalizeTransactionCount(self.transactionCount)

    #"m": true, "l": 6484065,"M": true,"q": "44113.00000000","a": 5378484,"T": 1591976004949,"p": "0.00000225","f": 6484064
    def AddData(self, jsonIn):
        isSell = jsonIn["m"]
        power  = float(jsonIn["q"]) * float(jsonIn["p"])
        if self.firstPrice == 0.0:
            self.firstPrice = float(jsonIn["p"])
        self.lastPrice = float(jsonIn["p"])
        self.totalTransactionCount += 1
        if not isSell:
            self.transactionCount += 1
            self.totalBuy += power
        else:
            self.totalSell += power

    def SetTime(self, timeInSecs):
        self.timeInSecs = timeInSecs

    def Reset(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.totalTransactionCount = 0.0
        self.score = 0
        self.timeInSecs = 0
        self.firstPrice = 0.0
        self.lastPrice = 0.0

class TransactionPattern:
    def __init__(self):
        self.transactionList = []
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0.0
        self.maxNormalizedCount = 0
        self.totalTransactionCount = 0
        self.timeDiffInSeconds = 0
        self.priceRatio = 1.0

    def Append( self, dataList, peakTime ):
        if len(dataList) > 0:
            self.priceRatio = int((1.0 - dataList[-1].lastPrice / dataList[0].firstPrice)*1000)

        for elem in dataList:
            elem.NormalizeTransactionCount()
            self.transactionList.append(elem.normalizedCount)
            self.maxNormalizedCount = max( self.maxNormalizedCount, elem.normalizedCount)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionCount
            self.totalTransactionCount += elem.totalTransactionCount
        self.timeDiffInSeconds = dataList[-1].timeInSecs - peakTime

    def TotalTrade(self):
        total = (self.totalBuy + self.totalSell) // 2

        if total > 10:
            total = 10
        return total

    def GetFeatures(self):
        returnval = self.transactionList + [self.totalTransactionCount,self.priceRatio,self.BuyVsSellRatio(),self.TotalTrade()]
        return returnval


    def BuyVsSellRatio(self):
        if self.totalSell == 0.0:
            return 4
        buySellRatio = self.totalBuy//self.totalSell
        return min(4,buySellRatio)

    def __repr__(self):
        return "list:%s,buySellRatio:%d,timeDiff:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f, maxNormalizedCount:%d" % (
                                                                                str(self.transactionList), self.BuyVsSellRatio(), self.timeDiffInSeconds,
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
    stopTime = 25

    def __init__(self, jsonIn, lowestAcceptedTotalTransactionCount, mseconds, isBottom, curveVal, curveTime, riseList, timeList ):
        self.mseconds = mseconds
        totalSize = len(jsonIn)
        self.patternList = []
        self.badPatternList = []
        self.dataList = []
        self.isBottom = isBottom
        self.curveVal = curveVal
        self.curveTime = curveTime
        self.inputRise = riseList
        self.inputTime = timeList
        self.lowestAcceptedTotalTransactionCount = lowestAcceptedTotalTransactionCount

        prices = list(map( lambda x: float(x["p"]), jsonIn))
        if len(prices) == 0:
            return

        if isBottom:
            self.peakIndex = prices.index(min(prices))
        else:
            self.peakIndex = prices.index(max(prices))

        self.peakVal = float(jsonIn[self.peakIndex]["p"])
        self.peakTimeSeconds = int(jsonIn[self.peakIndex]["T"])//1000
        self.startIndex = max(0, self.__FindIndexWithPriceAndPercent(jsonIn, self.peakIndex, -1, -1))
        self.stopIndex = min(totalSize, self.__FindIndexWithPriceAndPercent(jsonIn, self.peakIndex, totalSize,1))
        self.__DivideDataInSeconds(jsonIn)

    def GetTransactionPatterns(self):
        transactionPatterns = self.patternList
        returnVal = []
        for transactionPattern in transactionPatterns:
            returnVal.append( transactionPattern.GetFeatures() )
        return returnVal

    #TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
    def AssignScores( self, ngramCount ):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        #print(lenArray, self.dataList)
        for x in range( 0, lenArray ):
            curElement = self.dataList[x]
            curElement.NormalizeTransactionCount()
            self.__AppendToPatternList(ngramCount, x, lenArray)

    def __FindIndexWithPriceAndPercent(self, jsonIn, curStartIndex, curStopIndex, step):
        for x in range(curStartIndex, curStopIndex, step ):
            curElem = jsonIn[x]
            price = float(curElem["p"])
            curTime = int(curElem["T"])//1000

            if self.isBottom:
                if price > self.peakVal * (1.00+self.percent*2):
                    return x
            else:
                if price < self.peakVal * (1.00-self.percent*2):
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
                transactionData.SetTime(curMiliSecs//1000)
            else:
                transactionData.AddData(curElement)
        self.dataList.append(copy.deepcopy(transactionData))

    def __AppendToPatternList(self, ngramCount, curIndex, lenArray):
        startBin = curIndex + 1 - ngramCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        pattern = TransactionPattern()
        pattern.Append(self.dataList[startBin:endBin], self.peakTimeSeconds)
        if pattern.totalTransactionCount < self.lowestAcceptedTotalTransactionCount:
            return
        if pattern.maxNormalizedCount < 4:
            if sum( y > 1 for y in pattern.transactionList ) < 3 :
                return
        if self.isBottom:
            self.patternList.append(pattern)
        else:
            self.badPatternList.append(pattern)


class TransactionAnalyzer :
    TransactionCountPerSecBase = 40
    TransactionCountPerSecIncrease = 3

    def __init__(self ):
        self.featureArr = []
        self.patternList = []
        self.badPatternList = []
        self.peakHelperList = []

    def AddPeak(self, jsonIn, riseMinuteList, msec, ngrams, maxGrams ):
        for index in range(len(jsonIn)):
            isBottom = riseMinuteList[index].rise < 0.0
            if len(jsonIn[index]) == 0:
                continue
            indexPlusOne = index + 1
            riseList = list(map( lambda x: x.rise, riseMinuteList[index-maxGrams:indexPlusOne] ))
            timeList = list(map(lambda x: x.time, riseMinuteList[index-maxGrams:indexPlusOne]))
            if len(riseList) < maxGrams :
                riseList = [0.0]*(maxGrams-len(riseList)) + riseList
                timeList = [0.0]*(maxGrams-len(timeList)) + timeList

            totalSec = msec*ngrams//1000
            lowestTransaction = TransactionAnalyzer.TransactionCountPerSecBase + TransactionAnalyzer.TransactionCountPerSecIncrease * totalSec
            peakHelper = TransactionPeakHelper(jsonIn[index], lowestTransaction, msec, isBottom, riseMinuteList[index].rise, riseMinuteList[index].time, riseList, timeList)
            peakHelper.AssignScores(ngrams)
            self.peakHelperList.append(peakHelper)
            self.__MergeInTransactions(peakHelper,riseList,timeList)

    def toTransactionNumpy(self, ngrams ):
        goodSize = len(self.patternList)
        badSize = len(self.badPatternList)
        if (goodSize < badSize*0.75):
            upsampleSize = int(badSize*0.75) - goodSize
            print("I will upsample good: ", goodSize, " bad: ", badSize, " upsample: ", upsampleSize)
            for i in range(upsampleSize):
                self.patternList.append(self.patternList[goodSize-1-(i%goodSize)])
        allData = self.patternList + self.badPatternList

        self.featureArr = np.array(allData)
        self.featureArr.reshape(-1, ngrams+6)
        return self.featureArr

    def toTransactionCurves(self, ngrams):
        returnVal = []
        for peakHelper in self.peakHelperList:
                returnVal.append( peakHelper.inputRise[-ngrams:] )

        return returnVal


    def toTransactionCurvesToNumpy(self, ngrams):
        returnVal = []
        for peakHelper in self.peakHelperList:
            returnVal.append(peakHelper.inputRise[-ngrams:] + peakHelper.inputTime[-ngrams:])

        self.featureArr = np.array(returnVal)
        self.featureArr.reshape(-1, ngrams*2)
        return self.featureArr


    def toTransactionResultsNumpy(self):
        print(len(self.patternList), " ", len(self.badPatternList))
        goodResult = [1]*len(self.patternList)
        badResult = [0] * len(self.badPatternList)
        returnPatternList = goodResult + badResult

        return np.array(returnPatternList)

    def toTransactionPeakResultsNumpy(self):
        returnPatternList = []
        for peakHelper in self.peakHelperList:
            returnPatternList.append( 1.0 if peakHelper.isBottom else 0.0)

        return np.array(returnPatternList)


    def Print( self ):
        peakPatternValues = list(self.patternList)
        peakPatternValues.sort()
        m = sum(peakPatternValues) / len(peakPatternValues)
        var_res = sum((xi - m) ** 2 for xi in peakPatternValues) / len(peakPatternValues)
        print( " len: ", len(peakPatternValues), " small: ", peakPatternValues[0],
               " last: ", peakPatternValues[-1], " mean ", m, " var ", var_res)

    def __MergeInTransactions(self, transactionPeakHelper,riseList,timeList ):
        # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
        for pattern in transactionPeakHelper.patternList:
            self.patternList.append(pattern.GetFeatures() + [abs(transactionPeakHelper.curveVal),float(transactionPeakHelper.curveTime)])

        for pattern in transactionPeakHelper.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + [abs(transactionPeakHelper.curveVal),float(transactionPeakHelper.curveTime)])
