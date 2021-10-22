import json
from builtins import list
from datetime import datetime
import bisect
import copy
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import MarketStateManager
import TransactionBasics

PeakFeatureCount = TransactionBasics.PeakFeatureCount
MaxMinDataLen = 2
UpSideDataLen = 2
TotalExtraFeatureCount = PeakFeatureCount + MaxMinDataLen + UpSideDataLen
percent = 0.01


def GetStartIndex(jsonIn):
    for index in range(len(jsonIn)):
        if len(jsonIn[index]) == 0:
            continue
        return index
    return -1

class RiseMinute:
    def __init__(self, riseAndTimeStr):
        riseAndTimeStrPair = riseAndTimeStr.split("|")
        self.rise = float(riseAndTimeStrPair[0])
        self.time = riseAndTimeStrPair[1]

    def __repr__(self):
        return "Rise:%f,Time:%s" % (self.rise, self.time)



class PeakHandler:
    def __init__(self, jsonIn, isBottom, transactionParam, riseList, timeList):
        self.patternList = []
        self.mustBuyList = []
        self.badPatternList = []
        self.dataList = []
        self.riseList = riseList
        self.timeList = timeList
        self.marketStates = []
        self.isFinalize = True
        self.transactionParam = transactionParam
        totalSec = transactionParam.msec * transactionParam.gramCount / 1000
        self.lowestTransaction = TransactionBasics.TransactionCountPerSecBase
        self.acceptedTransLimit = TransactionBasics.TransactionLimitPerSecBase

        prices = list(map(lambda x: float(x["p"]), jsonIn))
        priceLen = len(prices)
        if priceLen == 0:
            return

        self.isBottom = isBottom

        if self.isBottom:
            self.peakIndex = prices.index(min(prices))
        else:
            self.peakIndex = prices.index(max(prices))
        self.peakVal = float(jsonIn[self.peakIndex]["p"])
        self.peakTimeSeconds = int(jsonIn[self.peakIndex]["T"]) // 1000
        self.__DivideDataInSeconds(jsonIn)
        self.__AssignScores()

    def GetFeatures(self):
        return TransactionBasics.GetMaxMinListWithTime(self.riseList, self.peakTimeSeconds,self.peakVal)

    # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
    def __AssignScores(self):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        # print(lenArray, self.dataList)
        for x in range(0, lenArray):
            self.__AppendToPatternList(self.transactionParam.gramCount, x, lenArray)
        del self.dataList

    def __AppendToPatternList(self, ngramCount, curIndex, lenArray):
        totalCount = TransactionBasics.GetTotalPatternCount(ngramCount)
        startBin = curIndex + 1 - totalCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        lastTotalTradePower = self.dataList[curIndex].totalBuy + self.dataList[curIndex].totalSell
        if self.dataList[curIndex].totalTransactionCount < self.lowestTransaction or \
            lastTotalTradePower < self.acceptedTransLimit:
            return
        totalElement = TransactionBasics.TotalElementLimitMsecs // self.transactionParam.msec
        totalTradePower = TransactionBasics.LastNElementsTransactionPower(self.dataList, curIndex, totalElement)
        if totalTradePower < TransactionBasics.TotalPowerLimit:
            return
        pattern = TransactionBasics.TransactionPattern()
        copyList = copy.deepcopy(self.dataList[startBin:endBin])
        dataRange = TransactionBasics.ReduceToNGrams(copyList, ngramCount)
        pattern.Append(dataRange, self.peakTimeSeconds, self.peakVal, None )
        pattern.SetPeaks( self.riseList, self.timeList )
        if timeList[-1] < 15:
            return
        if self.__GetCategory(curIndex) == 0:
            self.mustBuyList.append(pattern)
        elif self.__GetCategory(curIndex) == 1:
            self.patternList.append(pattern)
        elif self.__GetCategory(curIndex) == 2:
            self.badPatternList.append(pattern)


    def __GetCategory(self, curIndex):
        price = self.dataList[curIndex].lastPrice
        time = self.dataList[curIndex].timeInSecs

        if self.isBottom:
            if price < self.peakVal * 1.01:
                return 1  # Good
            else:
                return 2  # Bad
        else:
            if price < self.peakVal * 0.97 and time < self.peakTimeSeconds:
                return 1  # Good
            else:
                return 2
        return -1

    def __DivideDataInSeconds(self, jsonIn):
        transactionData = TransactionBasics.TransactionData()
        lastEndTime = 0
        stopMiliSecs = int(jsonIn[-1]["T"])
        for x in range(len(jsonIn)):
            curElement = jsonIn[x]
            curMiliSecs = int(curElement["T"])
            if x == 0:
                lastEndTime = curMiliSecs + self.transactionParam.msec

            if curMiliSecs > lastEndTime:
                copyData = copy.deepcopy(transactionData)
                self.dataList.append(copyData)
                transactionData.Reset()
                transactionData.AddData(curElement)
                transactionData.SetTime(curMiliSecs // 1000)
                lastEndTime += self.transactionParam.msec
                while True:
                    if curMiliSecs > lastEndTime and lastEndTime < stopMiliSecs:
                        lastEndTime += self.transactionParam.msec
                        emptyData = TransactionBasics.TransactionData()
                        emptyData.SetTime(lastEndTime // 1000)
                        emptyData.lastPrice = copyData.lastPrice
                        self.dataList.append(emptyData)
                    else:
                        break
            else:
                transactionData.AddData(curElement)
        self.dataList.append(copy.deepcopy(transactionData))

class PeakMerger:

    def __init__(self, transactionParam):
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []
        self.handlerList = []
        self.peakHelperList = []
        self.transactionParam = transactionParam

    def AddFile(self, jsonIn):
        for index in range(len(jsonIn)):
            if not jsonIn[index]:
                continue
            peakData = jsonIn[index]["peak"]
            riseAndTimeStrList = peakData.split(",")
            if len(riseAndTimeStrList) < 2:
                continue

            transactionData = jsonIn[index]["transactionList"]
            peakSize = len(riseAndTimeStrList)
            transactionDataSize = len(transactionData)
            if peakSize != transactionDataSize:
                print(peakSize, " ", transactionDataSize)
                return
            riseAndTimeList = []
            skipIndexes = []
            for x in range(peakSize):
                riseMinute = RiseMinute(riseAndTimeStrList[x])
                if len(riseAndTimeList) > 0 and riseAndTimeList[-1].rise * riseMinute.rise > 0:
                    skipIndexes.append(x - 1)
                    riseAndTimeList[-1] = riseMinute
                else:
                    riseAndTimeList.append(riseMinute)

            newTransParams = []
            for i in range(len(transactionData)):
                if i in skipIndexes:
                    continue
                newTransParams.append(transactionData[i])

            transactionData = newTransParams

            startIndex = GetStartIndex(transactionData)
            if startIndex == -1:
                return

            riseList = list(map(lambda x: float(x.rise), riseAndTimeList))
            timeList = list(map(lambda x: float(x.time), riseAndTimeList))

            for index in range(startIndex, len(transactionData)):
                if index < 8:
                    continue
                if PeakFeatureCount > 0:
                    curPriceList = riseList[index + 1 - PeakFeatureCount:index + 1]
                    curTimeList = timeList[index+1-PeakFeatureCount:index+1]
                else:
                    curPriceList = []
                    curTimeList = []
                isBottom = riseList[index] < 0
                handler = PeakHandler(transactionData[index], isBottom, self.transactionParam, curPriceList, curTimeList)
                self.handlerList.append(handler)

    def Finalize(self):
        for peak in self.handlerList:
            self.__MergeInTransactions(peak)
            del peak
        #self.Print(index)

    def toTransactionFeaturesNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #mustBuyCount = len(self.mustBuyList)
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

    def __MergeInTransactions(self, handler):
        for pattern in handler.patternList:
            self.patternList.append(pattern.GetFeatures() + handler.GetFeatures())

        for pattern in handler.mustBuyList:
            self.mustBuyList.append(pattern.GetFeatures() + handler.GetFeatures())

        for pattern in handler.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + handler.GetFeatures())

class PeakManager:

    def __init__(self, transactionParamList):
        self.transParamList = transactionParamList
        self.peakMergerList = []
        self.CreateHandlers()
        self.FeedMergers()
        self.FinalizeMergers()

    def FeedMergers(self):
        jumpDataFolderPath = os.path.abspath(os.getcwd()) + "/Data/CompleteData/"
        onlyJumpFiles = [f for f in listdir(jumpDataFolderPath) if isfile(join(jumpDataFolderPath, f))]
        for fileName in onlyJumpFiles:
            print("Reading Jump", jumpDataFolderPath + fileName, " ")
            file = open(jumpDataFolderPath + fileName, "r")
            try:
                jsonDictionary = json.load(file)
                for merger in self.peakMergerList:
                    merger.AddFile(jsonDictionary)
            except:
                print("There was a exception in ", fileName)

    def toTransactionFeaturesNumpy(self, index):
        return self.peakMergerList[index].toTransactionFeaturesNumpy()

    def toTransactionResultsNumpy(self, index):
        return self.peakMergerList[index].toTransactionResultsNumpy()

    def FinalizeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            self.peakMergerList[transactionIndex].Finalize()

    def CreateHandlers(self):
        for transactionIndex in range(len(self.transParamList)):
            newMerger = PeakMerger(self.transParamList[transactionIndex])
            self.peakMergerList.append(newMerger)