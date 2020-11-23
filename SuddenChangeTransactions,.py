import json
import datetime
import bisect
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import MarketStateManager
import TransactionBasics



class SuddenChangeHandler:
    percent = 0.01
    TotalFeatureCount = 0

    def __init__(self, jsonIn, transactionParam):
        self.jumpTimeInSeconds = 0
        self.reportTimeInSeconds = 0
        self.reportPrice = 0.0
        self.jumpPrice = 0.0
        self.transactions = []
        self.maxMinList = []
        self.riseList = []
        self.timeList = []
        self.currencyName = ""
        self.isRise = False
        self.downUpList = []
        self.transactionParam = transactionParam
        self.__Parse(jsonIn)

    def GetFeatures(self):
        return self.maxMinList + self.marketStates + self.inputTime[-3:] + self.peakRatios
            #self.maxMinList #+ self.marketStates + self.inputTime[-3:] + self.scoreList + self.peakRatios

    def Finalize(self):
        return

    def __Parse(self, jsonIn):
        self.isRise = bool(jsonIn["isRise"])
        self.jumpPrice = float(jsonIn["jumpPrice"])
        self.reportPrice = float(jsonIn["reportPrice"])
        self.reportTimeInSeconds = datetime.strptime(jsonIn["reportTime"], '%Y-%b-%d %H:%M:%S').total_seconds()
        self.riseList = jsonIn["riseList"]
        self.timeList = jsonIn["timeList"]
        self.maxMinList = jsonIn["maxMin"]
        self.jumpTimeInSeconds = datetime.strptime(jsonIn["time"], '%Y-%b-%d %H:%M:%S').total_seconds()
        self.downUpList = jsonIn["downUps"]
        self.currencyName = jsonIn["name"]
        self.transactions = jsonIn["transactions"]

    def __DivideDataInSeconds(self, jsonIn):
        transactionData = TransactionBasics.TransactionData()
        lastEndTime = 0
        stopMiliSecs = int(jsonIn[-1]["T"])
        for x in range(len(jsonIn)):
            curElement = jsonIn[x]
            curMiliSecs = int(curElement["T"])
            if x == 0:
                lastEndTime = curMiliSecs + self.transactions.mseconds

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
                        emptyData = TransactionBasics.TransactionData()
                        emptyData.SetTime(lastEndTime // 1000)
                        emptyData.lastPrice = copyData.lastPrice
                        self.dataList.append(emptyData)
                    else:
                        break
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
        if not self.isJumpInWindow:
            return -1
        price = self.dataList[curIndex].lastPrice
        firstPrice = self.dataList[curIndex].firstPrice
        time = self.dataList[curIndex].timeInSecs

        if self.isBottom:
            if price < self.peakVal * 1.01:
                return 1  # Good
            elif price < self.peakVal * 1.015 and time < self.peakTimeSeconds:
                return 1  # Good
        else:
            if price > self.peakVal * 0.98:
                return 2
            if price > self.peakVal * 0.97 and time > self.peakTimeSeconds:
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


class TransactionAnalyzer:
    TransactionCountPerSecBase = 20
    TransactionCountPerSecIncrease = 0.5
    TransactionLimitPerSecBase = 1.0
    TransactionLimitPerSecBaseIncrease = 0.02

    def __init__(self, ngrams):
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []
        self.peakHelperList = []
        self.ngrams = ngrams

    def AddFile(self, jsonIn):
        for index in range(len(jsonIn)):
            if len(jsonIn[index]) == 0:
                continue

            peakHelper = TransactionPeakHelper()
            peakHelper.AssignScores()

            self.peakHelperList.append(peakHelper)

    def Finalize(self, index):
        for peak in self.peakHelperList:
            self.__MergeInTransactions(peak)
            del peak
        #self.Print(index)

    def toTransactionNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #mustBuyCount = len(self.mustBuyList)
        totalGoodCount = goodCount #+ mustBuyCount
        #if badCount / totalGoodCount > 3:
        #    self.badPatternList = self.badPatternList[-(totalGoodCount * 3):]
        allData = np.concatenate( (self.patternList, self.badPatternList), axis=0)
        print("Good count: ", goodCount, " Bad Count: ", badCount)
        return allData

    def toTransactionResultsNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #mustBuyCount = len(self.mustBuyList)
        print("Good count: ", goodCount, " Bad Count: ", badCount)
        #mustBuyResult = [2] * mustBuyCount
        goodResult = [1] * goodCount
        badResult = [0] * len(self.badPatternList)
        returnPatternList = goodResult + badResult
        return returnPatternList

    def __MergeInTransactions(self, transactionPeakHelper):

        if transactionPeakHelper.isFinalize == False:
            return
        for pattern in transactionPeakHelper.patternList:
            self.patternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.mustBuyList:
            self.mustBuyList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())
