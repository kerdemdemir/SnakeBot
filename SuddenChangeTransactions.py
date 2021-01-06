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
import random


PeakFeatureCount = TransactionBasics.PeakFeatureCount
MaxMinDataLen = 8
UpSideDataLen = 6
TotalExtraFeatureCount = PeakFeatureCount + MaxMinDataLen + UpSideDataLen
percent = 0.01
IsOneFileOnly = False


class SuddenChangeHandler:
    WeirdCount = 0
    NormalCount = 0
    riseTotalPower = []
    fallTotalPower = []
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

        self.mustSellList = []
        self.keepList = []
        self.addedCount  = 0

        self.jumpState = []
        self.__Parse(jsonIn)



        self.lowestTransaction = TransactionBasics.TransactionCountPerSecBase
        self.acceptedTransLimit = TransactionBasics.TransactionLimitPerSecBase
        self.dataList = []
        tempTransaction = json.loads(jsonIn["transactions"])
        if len(tempTransaction) == 0:
            return
        lastTimeInSeconds = int(tempTransaction[-1]["T"]) // 1000
        if self.reportTimeInSeconds - lastTimeInSeconds > 1500:
            self.jumpTimeInSeconds -= 3600
            self.reportTimeInSeconds -= 3600

        self.minIndex = 0
        self.maxIndex = 0
        self.maxPrice = 0
        self.minPrice = 1000
        for index in range(len(tempTransaction)):
            transaction = tempTransaction[index]
            curTimeInSeconds = int(transaction["T"]) // 1000
            curPrice = float(transaction["p"])
            if curTimeInSeconds < self.jumpTimeInSeconds - 10 or curTimeInSeconds > self.reportTimeInSeconds+10:
                continue
            if curPrice <= self.minPrice:
                self.minPrice = curPrice
                self.minIndex = index
                self.minTime = curTimeInSeconds
            if curPrice >= self.maxPrice:
                self.maxPrice = curPrice
                self.maxIndex = index
                self.maxTime = curTimeInSeconds


        if self.isRise:
            if self.minTime > self.maxTime:
                if self.maxPrice/self.minPrice < 1.03:
                    return
                self.isRise = False
        else:
            if self.maxTime > self.minTime:
                if self.maxPrice/self.minPrice < 1.03:
                    return
                self.isRise = True

        if self.isRise:
            self.peakIndex = self.minIndex
        else:
            self.peakIndex = self.maxIndex

        self.peakTime = int(tempTransaction[self.peakIndex]["T"])//1000
        self.peakVal = float(tempTransaction[self.peakIndex]["p"])

        self.__DivideDataInSeconds(tempTransaction) #populates the dataList with TransactionData
        self.__AppendToPatternList() # deletes dataList and populates mustBuyList, patternList badPatternList

    def GetFeatures(self):
        return TransactionBasics.GetMaxMinList( self.maxMinList )
        #self.timeList[-PeakFeatureCount:] + self.riseList[-PeakFeatureCount:]
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

        for i in range(len(self.riseList) - 1 ):
            if self.riseList[i]*self.riseList[i+1] > 0.0:
                TransactionBasics.RiseListSanitizer(self.riseList, self.timeList)


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
        copyData = copy.deepcopy(transactionData)
        self.dataList.append(copyData)

    def __AppendToPatternList(self):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        # print(lenArray, self.dataList)
        maxTradeVal = 0
        maxIndex = 0
        for x in range(lenArray):
            curTimeInSeconds = self.dataList[x].timeInSecs
            if  curTimeInSeconds > self.reportTimeInSeconds+10:
                continue
            lastTotalTradePower = self.dataList[x].totalBuy + self.dataList[x].totalSell
            if lastTotalTradePower > maxTradeVal  :
                maxTradeVal = lastTotalTradePower
                maxIndex = x
            self.__AppendToPatternListImpl(self.transactionParam.gramCount, x, lenArray)

        if len(self.patternList) > TransactionBasics.MaximumSampleSizeFromGoodPattern:
           self.patternList = self.patternList[-TransactionBasics.MaximumSampleSizeFromGoodPattern:]
        if len(self.badPatternList) > TransactionBasics.MaximumSampleSizeFromPattern:
            self.badPatternList = random.sample(self.badPatternList, TransactionBasics.MaximumSampleSizeFromPattern)

        if self.isRise:
            SuddenChangeHandler.riseTotalPower.append(maxTradeVal)
        else:
            SuddenChangeHandler.fallTotalPower.append(maxTradeVal)

        if len(self.mustSellList) > TransactionBasics.MaximumSampleSizeFromPattern:
            self.mustSellList = random.sample(self.mustSellList, TransactionBasics.MaximumSampleSizeFromPattern)
        if len(self.mustBuyList) > TransactionBasics.MaximumSampleSizeFromPattern:
            self.mustBuyList = random.sample(self.mustBuyList, TransactionBasics.MaximumSampleSizeFromPattern)

        del self.dataList

    def __AppendToPatternListImpl(self, ngramCount, curIndex, lenArray):
        totalCount = TransactionBasics.GetTotalPatternCount(ngramCount)
        startBin = curIndex + 1 - totalCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        lastTotalTradePower = self.dataList[curIndex].totalBuy + self.dataList[curIndex].totalSell
        if lastTotalTradePower < self.acceptedTransLimit/2:
            return

        pattern = TransactionBasics.TransactionPattern()
        copyList = copy.deepcopy(self.dataList[startBin:endBin])
        dataRange = TransactionBasics.ReduceToNGrams(copyList, ngramCount)
        pattern.Append( dataRange, self.jumpTimeInSeconds, self.jumpPrice, self.marketState)
        pattern.SetPeaks(self.riseList, self.timeList)
        if self.__GetCategorySell(curIndex) == 1:
             self.mustSellList.append(copy.deepcopy(pattern))
        elif self.__GetCategorySell(curIndex) == 2:
             self.keepList.append(copy.deepcopy(pattern))

        totalElement = TransactionBasics.TotalElementLimitMsecs // self.transactionParam.msec
        totalTradePower = TransactionBasics.LastNElementsTransactionPower(self.dataList, curIndex, totalElement)
        if totalTradePower < TransactionBasics.TotalPowerLimit:
            return

        if self.dataList[curIndex].totalTransactionCount < self.lowestTransaction:
            return
        if lastTotalTradePower < self.acceptedTransLimit:
            return
        #print(pattern.marketStateList)
        category = self.__GetCategory(curIndex)
        if category == 0:
            self.mustBuyList.append(pattern)
        elif category == 1:
            self.addedCount += 1
            self.patternList.append(pattern)
        elif category == 2:
            self.badPatternList.append(pattern)
            self.addedCount += 1

    def __GetCategory(self, curIndex):
        price = self.dataList[curIndex].lastPrice
        minVal = self.dataList[curIndex].minPrice
        maxVal = self.dataList[curIndex].maxPrice
        curTimeSecs = self.dataList[curIndex].timeInSecs

        if self.isRise:
            if minVal < self.peakVal * 1.0125:
                if self.maxTime - curTimeSecs < 3:
                    return -1
                return 1  # Good
            elif curTimeSecs < self.peakTime:
                return 2
        else:
            if maxVal > self.peakVal * 0.98:
                return 2

        return -1

    def __GetCategorySell(self, curIndex):
        price = self.dataList[curIndex].lastPrice
        minVal = self.dataList[curIndex].minPrice
        maxVal = self.dataList[curIndex].maxPrice
        time = self.dataList[curIndex].timeInSecs

        if self.isRise:
            if minVal < self.peakVal * 1.03:
                return 2  # We can keep

        else:
            if maxVal > self.peakVal * 0.995:
                return 1 # We need to sell now
        return -1

class SuddenChangeMerger:

    def __init__(self, transactionParam, marketState):
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []

        self.mustSellList = []
        self.keepList = []

        self.handlerList = []
        self.peakHelperList = []
        self.transactionParam = transactionParam
        self.marketState = marketState

    def AddFile(self, jsonIn):
        for index in range(len(jsonIn)):
            if not jsonIn[index]:
                continue

            jsonPeakTrans = jsonIn[index]
            isRising = jsonPeakTrans["riseList"][-1] > 1.0

            if isRising:
                continue

            handler = SuddenChangeHandler(jsonPeakTrans,self.transactionParam,self.marketState)
            self.handlerList.append(handler)

    def Finalize(self):
        for peak in self.handlerList:
            self.__MergeInTransactions(peak)
            del peak
        #self.Print(index)

    def toTransactionFeaturesNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #self.Print()
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
        badResult = [0] * badCount
        returnPatternList = goodResult + badResult
        return returnPatternList

    def toSellTransactions(self):
        mustSellCount = len(self.mustSellList)
        keepCount = len(self.keepList)
        #self.Print()
        #mustBuyCount = len(self.mustBuyList)
        allData = np.concatenate( (self.mustSellList, self.keepList), axis=0)
        #print(allData)
        print("Must sell count: ", mustSellCount, " Keep count: ", keepCount)
        return allData

    def toSellResultsNumpy(self):
        mustSellCount = len(self.mustSellList)
        keepCount = len(self.keepList)

        print("Must sell count: ", mustSellCount, " Keep count: ", keepCount)
        mustSellResult = [0] * keepCount
        keepResult  = [1] * mustSellCount
        returnPatternList = mustSellResult + keepResult
        return returnPatternList

    def Print(self):

        #mustBuyList = np.array(self.mustBuyList)
        buyList = np.array(self.patternList)
        badList = np.array(self.badPatternList)

        for i in range(len(self.patternList[0])):
            a = {'Good': buyList[:, -(i + 1)],
                 'Bad': badList[:, -(i + 1)]}
            #df = pd.DataFrame.from_dict( a, orient='index')
            #df = df.transpose()
            #df.plot.box()
            #mustBuyLegend = str(np.quantile(mustBuyList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(mustBuyList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(mustBuyList[:, -(i + 1)], 0.9))
            buyLegend = str(np.quantile(buyList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(buyList[:, -(i + 1)], 0.25)) + "," \
                        + str(np.quantile(buyList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(buyList[:, -(i + 1)], 0.75)) + "," + str(np.quantile(buyList[:, -(i + 1)], 0.9))
            badLegend = str(np.quantile(badList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(badList[:, -(i + 1)], 0.25)) +\
                        "," + str(np.quantile(badList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(badList[:, -(i + 1)], 0.75)) + "," + str(np.quantile(badList[:, -(i + 1)], 0.9))
            print(str(self.transactionParam.msec) ,"_" , str(i), "_" , buyLegend , " ", badLegend)
            #plt.savefig('Plots/' + str(self.transactionParam.msec) + "_" + str(i) + "_box.pdf")
            #plt.cla()
            #plt.clf()

        #plt.close()


    def __MergeInTransactions(self, handler):
        for pattern in handler.patternList:
            self.patternList.append(pattern.GetFeatures() + handler.GetFeatures())

        for pattern in handler.mustBuyList:
            self.mustBuyList.append(pattern.GetFeatures() + handler.GetFeatures())

        for pattern in handler.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + handler.GetFeatures())

        for pattern in handler.mustSellList:
            self.mustSellList.append(pattern.GetFeatures() )

        for pattern in handler.keepList:
            self.keepList.append(pattern.GetFeatures())


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
        badListArray = np.array(SuddenChangeHandler.fallTotalPower)
        goodListArray = np.array(SuddenChangeHandler.riseTotalPower)
        badLegend = str(np.quantile(badListArray, 0.1)) + ", " + str(np.quantile(badListArray, 0.25)) + " , ** " \
                    + str(np.quantile(badListArray, 0.5)) + " ** ," + str(np.quantile(badListArray, 0.75)) + " , " + str(
            np.quantile(badListArray, 0.9))
        goodLegend = str(np.quantile(goodListArray, 0.1)) + " , " + str(np.quantile(goodListArray, 0.25)) + \
                     " , ** " + str(np.quantile(goodListArray, 0.5)) + " ** , " + str(np.quantile(goodListArray, 0.75)) + " , " + str(
            np.quantile(goodListArray, 0.9))
        print(" Good results ", goodLegend)
        print(" Bad results ", badLegend)

    def FeedMarketState(self):
        jumpDataFolderPath = os.path.abspath(os.getcwd()) + "/Data/JumpData/"
        onlyJumpFiles = [f for f in listdir(jumpDataFolderPath) if isfile(join(jumpDataFolderPath, f))]
        riseCount = 0
        downCount = 0
        for fileName in onlyJumpFiles:
            print("Reading market state", jumpDataFolderPath + fileName, " ")
            file = open(jumpDataFolderPath + fileName, "r")
            epoch = datetime.utcfromtimestamp(0)
            try:
                jsonDictionary = json.load(file)
                for jsonIn in jsonDictionary:
                    if not jsonIn:
                        continue

                    tempTransaction = json.loads(jsonIn["transactions"])
                    if len(tempTransaction) == 0:
                        continue

                    datetime_object = datetime.strptime(jsonIn["reportTime"].split(".")[0], '%Y-%b-%d %H:%M:%S')
                    reportTimeInSeconds = (datetime_object - epoch).total_seconds()

                    lastTimeInSeconds = int(tempTransaction[-1]["T"]) // 1000
                    if reportTimeInSeconds - lastTimeInSeconds > 1500:
                        reportTimeInSeconds -= 3600
                    isRise = bool(jsonIn["isRise"])
                    if isRise:
                        riseCount += 1
                    else:
                        downCount += 1
                    self.marketState.add(isRise, reportTimeInSeconds)
            except:
                print("There was a exception in ", fileName)
            if IsOneFileOnly:
                break
        self.marketState.sort()
        print("Total rise: ", riseCount, " total down: ", downCount)

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
            if IsOneFileOnly:
                break

    def toTransactionFeaturesNumpy(self, index):
        return self.suddenChangeMergerList[index].toTransactionFeaturesNumpy()

    def toTransactionResultsNumpy(self, index):
        return self.suddenChangeMergerList[index].toTransactionResultsNumpy()

    def toSellTransactions(self, index):
        return self.suddenChangeMergerList[index].toSellTransactions()

    def toSellResultsNumpy(self, index):
        return self.suddenChangeMergerList[index].toSellResultsNumpy()

    def FinalizeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            self.suddenChangeMergerList[transactionIndex].Finalize()

    def CreateSuddenChangeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            newMerger = SuddenChangeMerger(self.transParamList[transactionIndex], self.marketState)
            self.suddenChangeMergerList.append(newMerger)