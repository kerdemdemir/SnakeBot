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
MaxMinDataLen = 8
UpSideDataLen = 6
TotalExtraFeatureCount = PeakFeatureCount + MaxMinDataLen + UpSideDataLen
percent = 0.01

class BuySellHandler:
    def __init__(self, jsonIn, transactionParam,marketState):
        self.marketState = marketState
        self.buyTimeInSeconds = 0
        self.buyPrice = 0.0
        self.sellPrice = 0.0
        self.sellTimeInSeconds = 0

        self.transactions = []
        self.currencyName = ""
        self.transactionParam = transactionParam
        self.mustSellList = []
        self.keepList = []
        self.__Parse(jsonIn)

        self.lowestTransaction = TransactionBasics.TransactionCountPerSecBase/3
        self.acceptedTransLimit = TransactionBasics.TransactionLimitPerSecBase/3
        self.dataList = []

        tempTransaction = json.loads(jsonIn["transactions"])
        if len(tempTransaction) < 60:
            return
        prices = list(map(lambda x: float(x["p"]), tempTransaction))

        self.peakIndex = prices.index(max(prices))
        self.peakTime = int(tempTransaction[self.peakIndex]["T"])//1000
        self.peakVal = float(tempTransaction[self.peakIndex]["p"])

        self.minPeakIndex = prices.index(min(prices))
        self.minPeakTime = int(tempTransaction[self.minPeakIndex]["T"])//1000
        self.minPeakVal = float(tempTransaction[self.minPeakIndex]["p"])


        self.__DivideDataInSeconds(tempTransaction) #populates the dataList with TransactionData
        self.__AppendToPatternList() # deletes dataList and populates mustBuyList, patternList badPatternList

    def __Parse(self, jsonIn):

        epoch = datetime.utcfromtimestamp(0)
        self.buyPrice = float(jsonIn["buyPrice"])
        datetime_object = datetime.strptime(jsonIn["buyTime"].split(".")[0], '%Y-%b-%d %H:%M:%S')
        self.buyTimeInSeconds = (datetime_object - epoch).total_seconds()
        self.sellPrice = float(jsonIn["sellPrice"])
        datetime_object = datetime.strptime(jsonIn["sellTime"].split(".")[0], '%Y-%b-%d %H:%M:%S')
        self.sellTimeInSeconds = (datetime_object - epoch).total_seconds()
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
        for x in range(0, lenArray):
            self.__AppendToPatternListImpl(self.transactionParam.gramCount, x, lenArray)
        del self.dataList

    def __AppendToPatternListImpl(self, ngramCount, curIndex, lenArray):
        totalCount = TransactionBasics.GetTotalPatternCount(ngramCount)
        startBin = curIndex + 1 - totalCount
        endBin = curIndex + 1
        if startBin < 0 or endBin > lenArray:
            return

        lastTotalTradePower = self.dataList[curIndex].totalBuy + self.dataList[curIndex].totalSell
        if lastTotalTradePower < self.acceptedTransLimit:
            return

        pattern = TransactionBasics.TransactionPattern()
        copyList = copy.deepcopy(self.dataList[startBin:endBin])
        dataRange = TransactionBasics.ReduceToNGrams(copyList, ngramCount)
        pattern.AppendWithOutPeaks( dataRange, self.marketState, self.buyPrice, self.buyTimeInSeconds)

        if self.__GetCategory(curIndex) == 1:
            self.mustSellList.append(copy.deepcopy(pattern))
        elif self.__GetCategory(curIndex) == 2:
            self.keepList.append(copy.deepcopy(pattern))


    def __GetCategory(self, curIndex):
        price = self.dataList[curIndex].maxPrice
        time = self.dataList[curIndex].timeInSecs

        if price > self.peakVal * 0.997:
            return 1  # Sell
        elif time > self.peakTime and price > self.peakVal * 0.995:
            return 1  # Sell
        elif time > self.buyTimeInSeconds and time < self.peakTime:
            return 2
        return -1

class BuyAnalyzeMerger:
    def __init__(self, transactionParam, marketState):
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
            handler = BuySellHandler(jsonPeakTrans,self.transactionParam,self.marketState)
            self.handlerList.append(handler)

    def Finalize(self):
        for peak in self.handlerList:
            self.__MergeInTransactions(peak)
            del peak

    def toSellTransactions(self):
        mustSellCount = len(self.mustSellList)
        keepCount = len(self.keepList)
        #self.Print()
        print("Must sell count: ", mustSellCount, " Keep count: ", keepCount)

        allData = np.concatenate( (self.mustSellList, self.keepList), axis=0)
        print("Must sell count: ", mustSellCount, " Keep count: ", keepCount)
        return allData

    def toSellResultsNumpy(self):
        mustSellCount = len(self.mustSellList)
        keepCount = len(self.keepList)

        print("Must sell count: ", mustSellCount, " Keep count: ", keepCount)
        mustSellResult = [1] * mustSellCount
        keepResult  = [0] * keepCount
        returnPatternList = mustSellResult + keepResult
        return returnPatternList

    def Print(self):

        #mustBuyList = np.array(self.mustBuyList)
        buyList = np.array(self.mustSellList)
        badList = np.array(self.keepList)

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

        for pattern in handler.mustSellList:
            self.mustSellList.append(pattern.GetFeatures())

        for pattern in handler.keepList:
            self.keepList.append(pattern.GetFeatures())


class BuyAnalyzeManager:

    def __init__(self, transactionParamList, marketStateIn):
        self.marketState = marketStateIn
        self.transParamList = transactionParamList
        self.buySellMergerList = []
        self.CreateBuyAnalyzeMergers()
        print(self.buySellMergerList)
        self.BuyAnalyzeMergers()
        self.FinalizeMergers()

    def BuyAnalyzeMergers(self):
        jumpDataFolderPath = os.path.abspath(os.getcwd()) + "/Data/BuySellData/"
        onlyJumpFiles = [f for f in listdir(jumpDataFolderPath) if isfile(join(jumpDataFolderPath, f))]
        for fileName in onlyJumpFiles:
            print("Reading Buy", jumpDataFolderPath + fileName, " ")
            file = open(jumpDataFolderPath + fileName, "r")
            try:
                jsonDictionary = json.load(file)
                for merger in self.buySellMergerList:
                    merger.AddFile(jsonDictionary)
            except:
                print("There was a exception in ", fileName)


    def toSellTransactions(self, index):
        return self.buySellMergerList[index].toSellTransactions()

    def toSellResultsNumpy(self, index):
        return self.buySellMergerList[index].toSellResultsNumpy()

    def FinalizeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            self.buySellMergerList[transactionIndex].Finalize()

    def CreateBuyAnalyzeMergers(self):
        for transactionIndex in range(len(self.transParamList)):
            newMerger = BuyAnalyzeMerger(self.transParamList[transactionIndex], self.marketState)
            self.buySellMergerList.append(newMerger)