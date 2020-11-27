import json
from builtins import list
from datetime import datetime
import bisect
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from os import listdir
from os.path import isfile, join
import MarketStateManager
import TransactionBasics

PeakFeatureCount = TransactionBasics.PeakFeatureCount
MaxMinDataLen = 8
UpSideDataLen = 6
TotalExtraFeatureCount = PeakFeatureCount + MaxMinDataLen + UpSideDataLen
percent = 0.01



class TransactionPeakHelper:
    def __init__(self, jsonIn, transactionParam, riseList, timeList):
        self.patternList = []
        self.mustBuyList = []
        self.badPatternList = []
        self.dataList = []
        self.inputRise = riseList
        self.inputTime = [float(x) for x in timeList]
        self.marketStates = []
        self.isFinalize = True

        totalSec = transactionParam.msec * transactionParam.gramCount / 1000
        self.lowestTransaction = SuddenChangeHandler.TransactionCountPerSecBase + SuddenChangeHandler.TransactionCountPerSecIncrease * totalSec
        self.acceptedTransLimit = SuddenChangeHandler.TransactionLimitPerSecBase + SuddenChangeHandler.TransactionLimitPerSecBaseIncrease * totalSec
        self.buyTransLimit = SuddenChangeHandler.TransactionBuyLimit + SuddenChangeHandler.TransactionLimitPerSecBaseIncrease * totalSec

        self.isJumpInWindow = False
        prices = list(map(lambda x: float(x["p"]), jsonIn))
        priceLen = len(prices)
        if priceLen == 0:
            return

        self.isBottom = True if riseList[-1] < 0.0 else False

        if self.isBottom:
            self.peakIndex = prices.index(min(prices))
        else:
            self.peakIndex = prices.index(max(prices))

        self.peakVal = float(jsonIn[self.peakIndex]["p"])
        self.peakTimeSeconds = int(jsonIn[self.peakIndex]["T"]) // 1000
        self.__DivideDataInSeconds(jsonIn)
        self.__AssignScores()

    def GetPeakFeatures(self):
        return []

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
        startBin = curIndex + 1 - ngramCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        pattern = TransactionBasics.TransactionPattern()
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
        time = self.dataList[curIndex].timeInSecs

        if self.isBottom:
            if price < self.peakVal * 1.002:
                return 1  # Good
            elif price < self.peakVal * 1.015 and time < self.peakTimeSeconds:
                return 1  # Good
        else:
            if price > self.peakVal * 0.98:
                return 2
            if price > self.peakVal * 0.97 and time > self.peakTimeSeconds:
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
                        emptyData = TransactionBasics.TransactionData()
                        emptyData.SetTime(lastEndTime // 1000)
                        emptyData.lastPrice = copyData.lastPrice
                        self.dataList.append(emptyData)
                    else:
                        break
            else:
                transactionData.AddData(curElement)
        self.dataList.append(copy.deepcopy(transactionData))



class SuddenChangeHandler:
    TransactionCountPerSecBase = 6
    TransactionCountPerSecIncrease = 0.25
    TransactionLimitPerSecBase = 0.4
    TransactionLimitPerSecBaseIncrease = 0.025
    TransactionBuyLimit = 3.0

    def __init__(self, jsonIn, transactionParam,marketState):
        self.marketState = marketState
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
        self.patternList = []
        self.mustBuyList = []
        self.badPatternList = []
        self.jumpState = []
        self.__Parse(jsonIn)

        totalSec = transactionParam.msec * transactionParam.gramCount / 1000
        self.lowestTransaction = SuddenChangeHandler.TransactionCountPerSecBase + SuddenChangeHandler.TransactionCountPerSecIncrease * totalSec
        self.acceptedTransLimit = SuddenChangeHandler.TransactionLimitPerSecBase + SuddenChangeHandler.TransactionLimitPerSecBaseIncrease * totalSec
        self.buyTransLimit = SuddenChangeHandler.TransactionBuyLimit + SuddenChangeHandler.TransactionLimitPerSecBaseIncrease * totalSec
        self.dataList = []
        tempTransaction = json.loads(jsonIn["transactions"])
        self.__DivideDataInSeconds(tempTransaction) #populates the dataList with TransactionData
        self.__AppendToPatternList() # deletes dataList and populates mustBuyList, patternList badPatternList

    def GetFeatures(self):
        return []#self.timeList[-PeakFeatureCount:] + self.riseList[-PeakFeatureCount:]
        #return self.maxMinList  + self.timeList[-SuddenChangeHandler.PeakFeatureCount:] + self.riseList[-SuddenChangeHandler.PeakFeatureCount:]

    def __Parse(self, jsonIn):
        epoch = datetime.utcfromtimestamp(0)

        self.isRise = bool(jsonIn["isRise"])
        self.jumpPrice = float(jsonIn["jumpPrice"])
        self.reportPrice = float(jsonIn["reportPrice"])
        datetime_object = datetime.strptime(jsonIn["reportTime"].split(".")[0], '%Y-%b-%d %H:%M:%S')
        self.reportTimeInSeconds = (datetime_object - epoch).total_seconds()
        self.riseList = jsonIn["riseList"]
        self.timeList = jsonIn["timeList"]
        self.maxMinList = jsonIn["maxMin"]
        datetime_object = datetime.strptime(jsonIn["time"].split(".")[0], '%Y-%b-%d %H:%M:%S')
        self.jumpTimeInSeconds = (datetime_object - epoch).total_seconds()
        self.downUpList = jsonIn["downUps"]
        self.currencyName = jsonIn["name"]



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

    def __AppendToPatternList(self):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        # print(lenArray, self.dataList)
        for x in range(0, lenArray):
            self.__AppendToPatternListImpl(self.transactionParam.gramCount, x, lenArray)
        del self.dataList

    def __AppendToPatternListImpl(self, ngramCount, curIndex, lenArray):
        startBin = curIndex + 1 - ngramCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        pattern = TransactionBasics.TransactionPattern()
        pattern.Append(self.dataList[startBin:endBin], self.jumpTimeInSeconds, self.jumpPrice, self.marketState)
        pattern.SetPeaks( self.riseList, self.timeList )
        if pattern.totalTransactionCount < self.lowestTransaction or pattern.totalBuy+pattern.totalSell < self.acceptedTransLimit:
            if  pattern.totalBuy+pattern.totalSell < self.buyTransLimit:
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

        if self.isRise:
            if price < self.jumpPrice * 1.005:
                return 1  # Good
            #elif price < self.jumpPrice * 1.01 and time < self.jumpTimeInSeconds:
                #return 1  # Good
        else:
            if price > self.jumpPrice * 0.99:
                return 2
            if price > self.jumpPrice * 0.98 and time > self.jumpTimeInSeconds:
                return 2
        return -1

class SuddenChangeMerger:

    def __init__(self, transactionParam, marketState):
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []
        self.handlerList = []
        self.transactionParam = transactionParam
        self.marketState = marketState

    def AddFile(self, jsonIn):
        for index in range(len(jsonIn)):
            if not jsonIn[index]:
                continue

            handler = SuddenChangeHandler(jsonIn[index],self.transactionParam,self.marketState)
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

class SuddenChangeManager:

    def __init__(self, transactionParamList):
        self.marketState = MarketStateManager.MarketStateManager()
        self.FeedMarketState()

        self.transParamList = transactionParamList
        self.suddenChangeMergerList = []
        self.CreateSuddenChangeMergers()
        print(self.suddenChangeMergerList)
        self.FeedChangeMergers()
        self.FinalizeMergers()

    def FeedMarketState(self):
        jumpDataFolderPath = os.path.abspath(os.getcwd()) + "/Data/JumpData/"
        onlyJumpFiles = [f for f in listdir(jumpDataFolderPath) if isfile(join(jumpDataFolderPath, f))]
        for fileName in onlyJumpFiles:
            print("Reading Jump", jumpDataFolderPath + fileName, " ")
            file = open(jumpDataFolderPath + fileName, "r")
            epoch = datetime.utcfromtimestamp(0)
            try:
                jsonDictionary = json.load(file)
                for jsonIn in jsonDictionary:
                    if not jsonIn:
                        continue
                datetime_object = datetime.strptime(jsonIn["reportTime"].split(".")[0], '%Y-%b-%d %H:%M:%S')
                totalSecs = (datetime_object - epoch).total_seconds()
                isRise = bool(jsonIn["isRise"])
                self.marketState.add(isRise, totalSecs)
            except:
                print("There was a exception in ", fileName)

    def FeedChangeMergers(self):
        jumpDataFolderPath = os.path.abspath(os.getcwd()) + "/Data/JumpData/"
        onlyJumpFiles = [f for f in listdir(jumpDataFolderPath) if isfile(join(jumpDataFolderPath, f))]
        for fileName in onlyJumpFiles:
            print("Reading Jump", jumpDataFolderPath + fileName, " ")
            file = open(jumpDataFolderPath + fileName, "r")
            try:
                jsonDictionary = json.load(file)
                for merger in self.suddenChangeMergerList:
                    merger.AddFile(jsonDictionary)
            except:
                print("There was a exception in ", fileName)

    def toTransactionFeaturesNumpy(self, index):
        return self.suddenChangeMergerList[index].toTransactionFeaturesNumpy()

    def toTransactionResultsNumpy(self, index):
        return self.suddenChangeMergerList[index].toTransactionResultsNumpy()

    def FinalizeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            self.suddenChangeMergerList[transactionIndex].Finalize()

    def CreateSuddenChangeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            newMerger = SuddenChangeMerger(self.transParamList[transactionIndex], self.marketState)
            self.suddenChangeMergerList.append(newMerger)
