import json
import datetime
import bisect
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import MarketStateManager

ExtraFeatureCount = 1
ExtraLongPriceStateCount = 12
ExtraMarketStateCount = 6

def GetMaxMinWithTime(riseMinuteList, curIndex, targetTime):
    totalTime = 0
    ratioToCurVal = 1.0
    maxRatio = 0.0
    minRatio = 100.0
    count = 0
    for index in range(curIndex, 0, -1):
        time = float(riseMinuteList[index].time)
        rise = float(riseMinuteList[index].rise)
        totalTime += time
        if totalTime > targetTime:
            if count == 0:
                return [1.0, 1.0, count]
            else:
                return [min(1.0,minRatio), max(1.0,maxRatio), count]
        count += 1
        ratioToCurVal -= rise / 100.0
        #print(totalTime, " ", rise, " ", ratioToCurVal)

        maxRatio = max(ratioToCurVal, maxRatio)
        minRatio = min(ratioToCurVal, minRatio)
    if count == 0:
        return [1.0, 1.0, count]
    else:
        return [min(1.0,minRatio), max(1.0,maxRatio), count]


class TransactionData:
    def __init__(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionBuyCount = 0.0
        self.totalTransactionCount = 0.0
        self.score = 0
        self.timeInSecs = 0
        self.firstPrice = 0.0
        self.lastPrice = 0.0

    def __repr__(self):
        return "TotalBuy:%f,TotalSell:%f,TransactionCount:%f,Score:%f,LastPrice:%f,Time:%d" % (
        self.totalBuy, self.totalSell,
        self.transactionBuyCount, self.score, self.lastPrice, self.timeInSecs)

    # "m": true, "l": 6484065,"M": true,"q": "44113.00000000","a": 5378484,"T": 1591976004949,"p": "0.00000225","f": 6484064
    def AddData(self, jsonIn):
        isSell = jsonIn["m"]
        power = float(jsonIn["q"]) * float(jsonIn["p"])
        if self.firstPrice == 0.0:
            self.firstPrice = float(jsonIn["p"])
        self.lastPrice = float(jsonIn["p"])
        self.totalTransactionCount += 1
        if not isSell:
            self.transactionBuyCount += 1
            self.totalBuy += power
        else:
            self.totalSell += power

    def SetTime(self, timeInSecs):
        self.timeInSecs = timeInSecs

    def Reset(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionBuyCount = 0.0
        self.totalTransactionCount = 0.0
        self.score = 0
        self.timeInSecs = 0
        self.firstPrice = 0.0
        self.lastPrice = 0.0


class TransactionPattern:
    def __init__(self):
        self.transactionBuyList = []
        self.transactionSellList = []
        self.transactionBuyPowerList = []
        self.transactionSellPowerList = []
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0.0
        self.totalTransactionCount = 0
        self.timeDiffInSeconds = 0
        self.priceRatio = 1.0

    def Append(self, dataList, peakTime):
        if len(dataList) > 0 and dataList[0].firstPrice != 0.0:
            self.priceRatio = dataList[-1].lastPrice / dataList[0].firstPrice

        for elem in dataList:
            self.transactionBuyList.append(elem.transactionBuyCount)
            self.transactionSellList.append(self.totalTransactionCount - elem.transactionBuyCount)
            self.transactionBuyPowerList.append(self.totalBuy)
            self.transactionSellPowerList.append(self.totalSell)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionBuyCount
            self.totalTransactionCount += elem.totalTransactionCount
        self.timeDiffInSeconds = dataList[-1].timeInSecs - peakTime

    def GetFeatures(self):
        returnList = []
        for i in range(len(self.transactionBuyList)):
            returnList.append(self.transactionBuyList[i])
            returnList.append(self.transactionSellList[i])
            returnList.append(self.transactionBuyPowerList[i])
            returnList.append(self.transactionSellPowerList[i])
        returnList.append(self.priceRatio)
        return returnList

    def __repr__(self):
        return "list:%s,timeDiff:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f" % (
            str(self.transactionBuyList), self.timeDiffInSeconds,
            self.totalBuy, self.totalSell,
            self.transactionCount, self.score)


class TransactionPeakHelper:
    percent = 0.01
    stopTime = 25
    PeakFeatureCount = 25
    LowestTransactionCount = 1
    AvarageCaseCounter = 0

    def __init__(self, jsonIn, lowestAcceptedTotalTransactionCount, acceptedTotalTransactionLimit,
                 mseconds, isBottom, curveVal, curveTime, riseList, timeList, maxMinList):
        self.mseconds = mseconds
        totalSize = len(jsonIn)
        self.patternList = []
        self.mustBuyList = []
        self.badPatternList = []
        self.dataList = []
        self.isBottom = isBottom
        self.curveVal = curveVal
        self.curveTime = curveTime
        self.inputRise = riseList
        self.inputTime = [float(x) for x in timeList]
        self.scoreList = []
        self.maxMinList = maxMinList
        self.marketStates = []
        self.marketState = 1

        self.lowestAcceptedTotalTransactionCount = lowestAcceptedTotalTransactionCount
        self.acceptedTotalTransactionLimit = acceptedTotalTransactionLimit
        prices = list(map(lambda x: float(x["p"]), jsonIn))
        if len(prices) == 0:
            return

        if isBottom:
            self.peakIndex = prices.index(min(prices))
        else:
            self.peakIndex = prices.index(max(prices))

        self.peakVal = float(jsonIn[self.peakIndex]["p"])
        self.peakTimeSeconds = int(jsonIn[self.peakIndex]["T"]) // 1000
        self.__DivideDataInSeconds(jsonIn)

    def GetPeakFeatures(self):
        return self.maxMinList + self.marketStates + self.inputTime[-3:] + self.scoreList

    def GetTransactionPatterns(self):
        transactionPatterns = self.patternList
        returnVal = []
        for transactionPattern in transactionPatterns:
            returnVal.append(transactionPattern.GetFeatures())
        return returnVal

    # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
    def AssignScores(self, ngramCount):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        # print(lenArray, self.dataList)
        for x in range(0, lenArray):
            self.__AppendToPatternList(ngramCount, x, lenArray)
        del self.dataList

    def __AppendToPatternList(self, ngramCount, curIndex, lenArray):
        startBin = curIndex + 1 - ngramCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        pattern = TransactionPattern()
        pattern.Append(self.dataList[startBin:endBin], self.peakTimeSeconds)
        if pattern.totalTransactionCount < self.lowestAcceptedTotalTransactionCount:
            if pattern.totalBuy+pattern.totalSell < self.acceptedTotalTransactionLimit:
                return

        if self.__GetCategory(curIndex) == 0:
            self.mustBuyList.append(pattern)
            # self.patternList.append(pattern)
        elif self.__GetCategory(curIndex) == 1:
            self.patternList.append(pattern)
        elif self.__GetCategory(curIndex) == 2:
            self.badPatternList.append(pattern)


    def __GetCategory(self, curIndex):
        price = self.dataList[curIndex].lastPrice
        firstPrice = self.dataList[curIndex].firstPrice
        time = self.dataList[curIndex].timeInSecs

        if self.isBottom:
            if price < self.peakVal * 1.005:
                return 0  # Must buy
            elif price < self.peakVal * 1.01:
                return 1  # Good
            elif price < self.peakVal * 1.02 and time < self.peakTimeSeconds:
                return 1  # Good
            elif price > self.peakVal * 1.04:
                return 2
            elif price > self.peakVal * 1.03 and time > self.peakTimeSeconds:
                return 2
        else:
            if price < self.peakVal * 0.95 and time < self.peakTimeSeconds:
                return 0  # Must buy
            elif price > self.peakVal * 0.98:
                return 2
        return -1

    def SetMarketState(self, marketStates):
        self.marketStates = marketStates
        for index in range(len(marketStates) // 2):
            ratio = marketStates[index] / max(2, marketStates[index + 1])
            if ratio < 0.66 and marketStates[index + 1] > 2:
                self.marketState = 0
            elif ratio > 2.0:
                self.marketState = 2
        self.marketState = 1

    def __DivideDataInSeconds(self, jsonIn):
        transactionData = TransactionData()
        lastEndTime = 0
        stopMiliSecs = int(jsonIn[-1]["T"])
        for x in range(len(jsonIn)):
            curElement = jsonIn[x]
            curMiliSecs = int(curElement["T"])
            if x == 0:
                lastEndTime = curMiliSecs + self.mseconds

            if curMiliSecs > lastEndTime:
                copyData = copy.deepcopy(transactionData)
                self.dataList.append(copyData)
                transactionData.Reset()
                transactionData.AddData(curElement)
                transactionData.SetTime(curMiliSecs // 1000)
                lastEndTime += self.mseconds
                while True:
                    if curMiliSecs > lastEndTime and lastEndTime < stopMiliSecs:
                        lastEndTime += self.mseconds
                        emptyData = TransactionData()
                        emptyData.SetTime(lastEndTime // 1000)
                        emptyData.lastPrice = copyData.lastPrice
                        self.dataList.append(emptyData)
                    else:
                        break
            else:
                transactionData.AddData(curElement)
        self.dataList.append(copy.deepcopy(transactionData))


class TransactionAnalyzer:
    TransactionCountPerSecBase = 20
    TransactionCountPerSecIncrease = 1
    TransactionLimitPerSecBase = 1.25
    TransactionLimitPerSecBaseIncrease = 0.05

    def __init__(self):
        self.featureArr = []
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []
        self.peakHelperList = []

    def GetStartIndex(self, jsonIn):
        for index in range(len(jsonIn)):
            if len(jsonIn[index]) == 0:
                continue
            return index
        return -1

    def AddCurrency(self, jsonIn, riseMinuteList, msec, ngrams, maxGrams, marketState):
        for index in range(len(jsonIn)):
            isBottom = riseMinuteList[index].rise < 0.0
            if len(jsonIn[index]) == 0:
                continue
            indexPlusOne = index + 1
            riseList = list(map(lambda x: x.rise, riseMinuteList[index - maxGrams:indexPlusOne]))
            timeList = list(map(lambda x: x.time, riseMinuteList[index - maxGrams:indexPlusOne]))
            if len(riseList) < maxGrams:
                riseList = [0.0] * (maxGrams - len(riseList)) + riseList
                timeList = [0.0] * (maxGrams - len(timeList)) + timeList

            #6 + 7 + 9 + 13 + 21 + 37 + 69
            maxMinList = GetMaxMinWithTime(riseMinuteList, index, 6*60) + GetMaxMinWithTime(riseMinuteList, index, 24*60) + \
                              GetMaxMinWithTime(riseMinuteList, index, 48*60)+ GetMaxMinWithTime(riseMinuteList, index, 72 * 60)
            #print(maxMinList)
            totalSec = msec * ngrams / 1000
            lowestTransaction = TransactionAnalyzer.TransactionCountPerSecBase + TransactionAnalyzer.TransactionCountPerSecIncrease * totalSec
            acceptedTransLimit = TransactionAnalyzer.TransactionLimitPerSecBase + TransactionAnalyzer.TransactionLimitPerSecBaseIncrease * totalSec

            peakHelper = TransactionPeakHelper(jsonIn[index], lowestTransaction, acceptedTransLimit, msec, isBottom,
                                               riseMinuteList[index].rise, riseMinuteList[index].time, riseList,
                                               timeList,maxMinList)
            peakHelper.AssignScores(ngrams)
            if marketState is not None:
                marketState.add(peakHelper)
            self.peakHelperList.append(peakHelper)

    def Finalize(self, index):
        for peak in self.peakHelperList:
            self.__MergeInTransactions(peak)
            del peak
        self.Print(index)

    def toTransactionNumpy(self, ngrams):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        mustBuyCount = len(self.mustBuyList)
        totalGoodCount = goodCount + mustBuyCount
        if badCount / totalGoodCount > 3:
            self.badPatternList = self.badPatternList[-(totalGoodCount * 3):]
        allData = self.patternList + self.mustBuyList + self.badPatternList
        print("Good count: ", goodCount, " Bad Count: ", badCount, " Must buy: ", mustBuyCount)
        self.featureArr = np.array(allData)
        self.featureArr.reshape(-1, ngrams * 4 + ExtraFeatureCount + TransactionPeakHelper.PeakFeatureCount)
        return self.featureArr

    def toTransactionResultsNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        mustBuyCount = len(self.mustBuyList)
        print("Good count: ", goodCount, " Bad Count: ", badCount, " Must buy: ", mustBuyCount)
        mustBuyResult = [2] * mustBuyCount
        goodResult = [1] * goodCount
        badResult = [0] * len(self.badPatternList)
        returnPatternList = goodResult + mustBuyResult + badResult
        return np.array(returnPatternList)

    def Print(self, index):

        columnNames = "PriceDiff, 6HMin,6HMax,6HPeakCount,24HMin,24HMax,24HPeakCount,48HMin,48HMax,48HPeakCount,72HMin,72HMax,72HPeakCount," \
                      "5MinsDowns,5MinsUp,30MinsDown,30MinsUp,60MinsDown,60MinsUp,Time1,Time2,Time3," \
                      "Score1,Score2,Score3,Score4"
        colNameList = columnNames.split(",")
        for i in range(TransactionPeakHelper.PeakFeatureCount):
            df = pd.DataFrame(
                {'Must': self.mustBuyList[-(i+1)], 'Good': self.patternList[-(i+1)], 'Bad': self.badPatternList[-(i+1)]})
            df.plot.hist(bins=100)
            plt.savefig('Plots/' + str(index) + "_" + str(i) + "_" + colNameList[-(i+1)] + '_histogram.pdf')
            plt.cla()

    def __PrintImpl(self, inList, extraMessage):
        peakPatternValues = np.array(inList);
        print(extraMessage," len: ", len(peakPatternValues), " mean ", np.mean(peakPatternValues, axis=0))

    def __MergeInTransactions(self, transactionPeakHelper):
        # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
        for pattern in transactionPeakHelper.patternList:
            self.patternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.mustBuyList:
            self.mustBuyList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())
