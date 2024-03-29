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
import random


PeakFeatureCount = TransactionBasics.PeakFeatureCount
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
            self.peakIndex = self.minIndex
        else:
            self.peakIndex = self.maxIndex

        self.peakTime = int(tempTransaction[self.peakIndex]["T"])//1000
        self.peakVal = float(tempTransaction[self.peakIndex]["p"])

        self.__DivideDataInSeconds(tempTransaction, self.transactionParam.msec, self.dataList, 0, len(tempTransaction)) #populates the dataList with TransactionData
        self.__AppendToPatternList(tempTransaction) # deletes dataList and populates mustBuyList, patternList badPatternList


    def GetFeatures(self):
        #return self.downUpList
        return TransactionBasics.GetMaxMinList( self.maxMinList )
        #return []
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
                self.riseList.pop(i)
                self.timeList.pop(i)
                #TransactionBasics.RiseListSanitizer(self.riseList, self.timeList)


        self.maxMinList = jsonIn["maxMin"]
        datetime_object = datetime.strptime(jsonIn["time"].split(".")[0], '%Y-%b-%d %H:%M:%S')
        self.jumpTimeInSeconds = (datetime_object - epoch).total_seconds()
        self.downUpList = jsonIn["downUps"]
        self.currencyName = jsonIn["name"]

    def __DivideDataInSeconds(self, jsonIn, msecs, datalist, startIndex, endIndex ):
        transactionData = TransactionBasics.TransactionData()
        lastEndTime = 0
        stopMiliSecs = int(jsonIn[endIndex-1]["T"])+1000
        for x in range(startIndex,endIndex):
            curElement = jsonIn[x]
            curMiliSecs = int(curElement["T"])
            if x == startIndex:
                lastEndTime = curMiliSecs//msecs*msecs + msecs
                transactionData.SetTime(curMiliSecs // msecs)

            if curMiliSecs > lastEndTime:
                copyData = copy.deepcopy(transactionData)
                datalist.append(copyData)
                transactionData.Reset()
                transactionData.AddData(curElement)

                while True:
                    if curMiliSecs > (lastEndTime + msecs) and lastEndTime < stopMiliSecs:
                        emptyData = TransactionBasics.TransactionData()
                        emptyData.SetTime(lastEndTime // msecs)
                        emptyData.SetIndex(x)
                        emptyData.lastPrice = copyData.lastPrice
                        lastEndTime += msecs
                        if startIndex == 0:
                            datalist.append(emptyData)
                    else:
                        transactionData.SetTime(curMiliSecs // msecs)
                        transactionData.SetIndex(x)
                        lastEndTime += msecs
                        break
            else:
                transactionData.AddData(curElement)
                transactionData.SetIndex(x)
        copyData = copy.deepcopy(transactionData)
        datalist.append(copyData)

    def __AppendToPatternList(self, jsonIn):
        lenArray = len(self.dataList)
        if lenArray == 0:
            return
        # print(lenArray, self.dataList)
        maxTradeVal = 0
        for x in range(lenArray):
            curTimeInSeconds = self.dataList[x].timeInSecs
            if  curTimeInSeconds > self.reportTimeInSeconds+10:
                continue
            lastTotalTradePower = self.dataList[x].totalBuy + self.dataList[x].totalSell
            if lastTotalTradePower > maxTradeVal  :
                maxTradeVal = lastTotalTradePower
            self.__AppendToPatternListImpl(self.transactionParam.gramCount, x, lenArray, jsonIn)
        if len(self.patternList) == 0:
            self.dataList.reverse()

        if len(self.patternList) > TransactionBasics.MaximumSampleSizeFromGoodPattern:
            sorted(self.patternList, key=lambda l: l.totalBuy)
            self.patternList = self.patternList[-TransactionBasics.MaximumSampleSizeFromGoodPattern:]

        if len(self.badPatternList) > TransactionBasics.MaximumSampleSizeFromPattern:
            self.badPatternList = random.sample(self.badPatternList, TransactionBasics.MaximumSampleSizeFromPattern)

        if self.isRise:
            SuddenChangeHandler.riseTotalPower.append(maxTradeVal)
        else:
            SuddenChangeHandler.fallTotalPower.append(maxTradeVal)

        del self.dataList

    def __AppendToPatternListImpl(self, ngramCount, curIndex, lenArray, jsonIn):
        totalCount = TransactionBasics.GetTotalPatternCount(ngramCount)
        startBin = curIndex + 1 - totalCount
        endBin = curIndex + 1
        if startBin < 0 or curIndex > lenArray:
             return

        curPattern = self.dataList[curIndex]
        lastTotalCount = curPattern.transactionBuyCount
        # if lastTotalCount < self.lowestTransaction:
        #     return

        if curPattern.totalBuy < 0.1:
            return

        # if self.dataList[curIndex].totalSell > 0.5:
        #     return
        #
        #if self.isRise:
        #    print("Analyzing rise")
        pattern = TransactionBasics.TransactionPattern()
        copyList = copy.deepcopy(self.dataList[startBin:endBin])
        dataRange = TransactionBasics.ReduceToNGrams(copyList, ngramCount)
        #
        #if dataRange[0].totalTransactionCount < 0.1:
        #    return
        # #

        if dataRange[0].transactionBuyCount < 0.09 or dataRange[0].transactionBuyCount > 1.2:
            return

        if dataRange[0].totalBuy < 0.0005 or dataRange[0].totalBuy > 0.012:
            return

        if dataRange[0].totalSell < 0.0002 or dataRange[0].totalSell > 0.008:
            return

        if dataRange[1].transactionBuyCount < 0.16 or dataRange[1].transactionBuyCount > 2.05:
            return

        if dataRange[1].totalBuy < 0.0012 or dataRange[1].totalBuy > 0.023:
            return

        if dataRange[1].totalSell < 0.00035 or dataRange[1].totalSell > 0.015:
            return

        if dataRange[2].transactionBuyCount < 0.45 or dataRange[2].transactionBuyCount > 3.02:
            return

        if dataRange[2].totalBuy < 0.003 or dataRange[2].totalBuy > 0.023:
            return

        if dataRange[2].totalSell < 0.00013 or dataRange[2].totalSell > 0.017:
            return


        # #
        # if dataRange[1].totalBuy < 0.0016:
        #     return
        #
        # if dataRange[1].totalSell > 0.01 and dataRange[1].totalSell > dataRange[1].totalBuy:
        #     return
        # elif dataRange[1].totalSell > 0.02:
        #     return
        #
        # if dataRange[0].totalSell > 0.005 and dataRange[0].totalSell > dataRange[0].totalBuy:
        #     return
        # elif dataRange[0].totalSell > 0.01:
        #     return
        #
        #
        # if dataRange[-1].totalSell > 0.01 and dataRange[-1].totalSell > dataRange[-1].totalBuy:
        #     return
        # elif dataRange[-1].totalSell > 0.1:
        #     return
        # #
        # # if dataRange[1].totalBuy/dataRange[0].totalBuy > 6.0:
        # #     return
        #
        if dataRange[0].transactionBuyCount > 0.0 and dataRange[-1].transactionBuyCount/dataRange[0].transactionBuyCount < 2.0:
            return
        # #
        if dataRange[0].totalBuy > 0.0 and dataRange[-1].totalBuy/dataRange[0].totalBuy < 7.0:
            return


        # if dataRange[0].totalBuy > 0.0 and dataRange[1].totalBuy/dataRange[0].totalBuy < 0.35:
        #     return
        #
        # if dataRange[0].totalBuy > 0.0 and dataRange[2].totalBuy/dataRange[0].totalBuy < 0.35:
        #     return

        if dataRange[0].firstPrice != 0.0:
            firstRatio = dataRange[0].lastPrice / dataRange[0].firstPrice
            if firstRatio < 1.004 or firstRatio > 1.055:
                return

        if dataRange[1].firstPrice != 0.0:
            firstRatio = dataRange[1].lastPrice / dataRange[1].firstPrice
            if firstRatio < 0.994 or firstRatio > 1.02:
                return

        if dataRange[2].firstPrice != 0.0:
            firstRatio = dataRange[2].lastPrice / dataRange[2].firstPrice
            if firstRatio < 0.99 or firstRatio > 1.016:
                return

        firstRatio = dataRange[-1].lastPrice / dataRange[-1].firstPrice
        if firstRatio < 0.998 or firstRatio > 1.01:
            return



        detailDataList = []
        self.__DivideDataInSeconds(jsonIn, 100, detailDataList, curPattern.startIndex-1, curPattern.endIndex+1)
        pattern.SetDetailedTransaction(detailDataList, dataRange)
        if pattern.maxDetailBuyPower < 0.045 or pattern.maxDetailBuyPower > 2.0:
            return

        basePrice = self.dataList[curIndex].lastPrice
        baseIndex = 0
        curIndexDetail = 0
        for detailPattern in detailDataList:
            if baseIndex == 0 and detailPattern.totalBuy > self.acceptedTransLimit:
                basePrice = detailPattern.lastPrice
                baseIndex = curIndexDetail
            curIndexDetail += 1

        totalSellCount = 0
        totalSellPower = 0.0
        for index in range(baseIndex+1, len(detailDataList)):
            totalSellPower += detailDataList[index].totalSell
            totalSellCount += (detailDataList[index].totalTransactionCount - detailDataList[index].transactionBuyCount)
        pattern.detailedHighestSellCountNumber = totalSellCount
        # if totalSellPower > 0.05:
        #     return

        # if not pattern.isMaxBuyLater:
        #     return
        if pattern.detailLen < 2:
            return
        #if self.timeList[-1] < 15:
        #    return


        ratio = basePrice/self.jumpPrice
        curTimeDiff = (self.dataList[curIndex].timeInSecs - self.jumpTimeInSeconds)//60
        pattern.timeToJump = self.reportTimeInSeconds - self.dataList[curIndex].timeInSecs
        pattern.SetPeaks(self.riseList, self.timeList, ratio, curTimeDiff)

        if pattern.totalPeakCount15M > 2.0:
            return
        if pattern.totalPeakCount1Hour > 5.0:
            return
        # if pattern.lastUpRatio < -1.0:
        #     return
        #
        # if pattern.peaks[-1] < 0.0 and pattern.lastDownRatio < -1.0:
        #     return

        reverseRatio = 1/ratio
        if self.maxMinList[0] * reverseRatio < 0.75 or self.maxMinList[0] * reverseRatio > 0.98:
            return

        if self.maxMinList[2] * reverseRatio < 0.65 or self.maxMinList[4] * reverseRatio < 0.55:
            return


        moreDetailDataList = []
        self.__DivideDataInSeconds(jsonIn, 1, moreDetailDataList, self.dataList[curIndex].startIndex, self.dataList[curIndex].endIndex+1)
        pattern.SetDetailedTransaction(moreDetailDataList, dataRange)
        # if pattern.detailLen < 5:
        #     return

        pattern.Append( dataRange, self.jumpTimeInSeconds, self.jumpPrice, self.marketState)

        #print(pattern.marketStateList)
        category = self.__GetCategory(curIndex,basePrice,pattern)
        if category == 0:
            self.mustBuyList.append(pattern)
        elif category == 1:
            self.addedCount += 1
            self.patternList.append(pattern)
        elif category == 2:
            self.badPatternList.append(pattern)
            self.addedCount += 1

    def __GetCategory(self, curIndex, priceIn, pattern):
        if self.isRise:
            if priceIn < self.reportPrice * 0.97:
                for i in range(curIndex+1, len(self.dataList)):
                    ratio = self.dataList[i].lastPrice / priceIn
                    timeDiff = self.dataList[i].endIndex - self.dataList[curIndex].endIndex
                    #pattern.UpdatePrice(timeDiff, ratio)
                    if ratio<0.98:
                        return -1
                    if ratio>1.03:
                        #pattern.GoalReached(timeDiff, 1.03)
                        return 1
                return -1
        else:
            for i in range(curIndex, len(self.dataList)):
                if self.dataList[i].lastPrice/priceIn<0.98:
                    return 2
                if self.dataList[i].lastPrice/priceIn>1.03:
                    return -1
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

            handler = SuddenChangeHandler(jsonPeakTrans,self.transactionParam,self.marketState)
            self.handlerList.append(handler)

    def Finalize(self):
        for peak in self.handlerList:
            self.__MergeInTransactions(peak)
            del peak
        self.Print()

    def toTransactionFeaturesNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #self.Print()
        #mustBuyCount = len(self.mustBuyList)
        print("Good count: ", goodCount, " Bad Count: ", badCount)

        allData = np.concatenate( (self.patternList, self.badPatternList), axis=0)
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
            # a = {'Good': buyList[:, i],
            #      'Bad': badList[:, i]}
            #df = pd.DataFrame.from_dict( a, orient='index')
            #df = df.transpose()
            #df.plot.box()
            #mustBuyLegend = str(np.quantile(mustBuyList[:, i], 0.1)) + "," + str(np.quantile(mustBuyList[:, i], 0.5)) + "," + str(np.quantile(mustBuyList[:, i], 0.9))
            #buyLegend = str(np.quantile(buyList[:, i], 0.1)) + "," + str(np.quantile(buyList[:, i], 0.25)) + "," +  str(np.quantile(buyList[:, i], 0.5)) + "," + str(np.quantile(buyList[:, i], 0.75)) + "," + str(np.quantile(buyList[:, i], 0.9))
            buyLegend = str(np.quantile(buyList[:, i], 0.1)) + "," + str(np.quantile(buyList[:, i], 0.9))

            if len(badList) > 0:
                badLegend = str(np.quantile(badList[:, i], 0.1)) + "," + str(np.quantile(badList[:, i], 0.25)) + "," +  str(np.quantile(badList[:, i], 0.5)) + "," + str(np.quantile(badList[:, i], 0.75)) + "," + str(np.quantile(badList[:, i], 0.9))
            else:
                badLegend = "empty"
            #print(str(self.transactionParam.msec) ,"_" , str(i), "_" , buyLegend , " ", badLegend)
            print(str(self.transactionParam.msec) ,"_" , str(i), "_" , buyLegend )

            #plt.savefig('Plots/' + str(self.transactionParam.msec) + "_" + str(i) + "_box.pdf")
            #plt.cla()
            #plt.clf()
        print("Good count: ", len(self.patternList))
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
        #self.FeedMarketState()

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
                    timeStr = jsonIn["reportTime"]
                    datetime_object = datetime.strptime(timeStr.split(".")[0], '%Y-%b-%d %H:%M:%S')
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
            except Exception as e:
                print("There was a exception in ", fileName, e)
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
            #try:
            jsonDictionary = json.load(file)
            for merger in self.suddenChangeMergerList:
                merger.AddFile(jsonDictionary)
            #except Exception as e:
            #    print("There was a exception in ", fileName, e )
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