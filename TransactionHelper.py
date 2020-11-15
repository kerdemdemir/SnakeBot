import json
import datetime
import bisect
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import MarketStateManager

ExtraFeatureCount = 0
ExtraPerDataInfo = 2 #MaxPriceMinPriceRatio
ExtraLongPriceStateCount = 8
ExtraMarketStateCount = 4

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
                return [1.0, 1.0]
            else:
                return [min(1.0,minRatio), max(1.0,maxRatio)]
        count += 1
        ratioToCurVal -= rise / 100.0
        #print(totalTime, " ", rise, " ", ratioToCurVal)

        maxRatio = max(ratioToCurVal, maxRatio)
        minRatio = min(ratioToCurVal, minRatio)
    if count == 0:
        return [1.0, 1.0]
    else:
        return [min(1.0,minRatio), max(1.0,maxRatio)]


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
        self.priceMaxRatio = 1.0
        self.priceMinRatio = 1.0

    def Append(self, dataList, peakTime):

        if len(dataList) > 0:
            priceList = list(map(lambda x: x.lastPrice, dataList))
            self.priceMaxRatio = dataList[-1].lastPrice / max(priceList)
            self.priceMinRatio = dataList[-1].lastPrice / min(priceList)

        for elem in dataList:
            self.transactionBuyList.append(elem.transactionBuyCount)
            self.transactionSellList.append(elem.totalTransactionCount - elem.transactionBuyCount)
            self.transactionBuyPowerList.append(elem.totalBuy)
            self.transactionSellPowerList.append(elem.totalSell)
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
        returnList.append(self.priceMinRatio)
        returnList.append(self.priceMaxRatio)
        return returnList

    def __repr__(self):
        return "list:%s,timeDiff:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f" % (
            str(self.transactionBuyList), self.timeDiffInSeconds,
            self.totalBuy, self.totalSell,
            self.transactionCount, self.score)


class TransactionPeakHelper:
    percent = 0.01
    stopTime = 25
    PeakFeatureCount = 21
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
            if price < self.peakVal * 1.006:
                return 1  # Good
            elif price < self.peakVal * 1.01 and time < self.peakTimeSeconds:
                return 1  # Good
        else:
            if price > self.peakVal * 0.97:
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
    TransactionCountPerSecIncrease = 0.5
    TransactionLimitPerSecBase = 1.0
    TransactionLimitPerSecBaseIncrease = 0.02

    def __init__(self, ngrams):
        self.mustBuyList = []
        self.patternList = []
        self.badPatternList = []
        self.peakHelperList = []
        self.ngrams = ngrams

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
        self.badPatternList = self.__trimExtremes(self.badPatternList)
        self.patternList = self.__trimExtremes(self.patternList)
        #self.mustBuyList = self.__trimExtremes(self.mustBuyList)
        #self.Print(index)

    def toTransactionNumpy(self):
        badCount = len(self.badPatternList)
        goodCount = len(self.patternList)
        #mustBuyCount = len(self.mustBuyList)
        totalGoodCount = goodCount #+ mustBuyCount
        if badCount / totalGoodCount > 3:
            self.badPatternList = self.badPatternList[-(totalGoodCount * 3):]
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

    def Print(self, index):

        #mustBuyList = np.array(self.mustBuyList)
        buyList = np.array(self.patternList)
        badList = np.array(self.badPatternList)

        columnNames = "PriceDiffMin,PriceDiffMax,6HMin,6HMax,24HMin,24HMax,48HMin,48HMax,72HMin,72HMax," \
                      "1MinsDowns,1MinsUp,5MinsDown,5MinsUp,Time1,Time2,Time3," \
                      "Score1,Score2,Score3,Score4"
        print("Printing index: ", index )
        colNameList = columnNames.split(",")
        for i in range(TransactionPeakHelper.PeakFeatureCount):
            a = {'Good': buyList[:, -(i + 1)],
                 'Bad': badList[:, -(i + 1)]}
            df = pd.DataFrame.from_dict( a, orient='index')
            df = df.transpose()
            df.plot.box()
            #mustBuyLegend = str(np.quantile(mustBuyList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(mustBuyList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(mustBuyList[:, -(i + 1)], 0.9))
            buyLegend = str(np.quantile(buyList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(buyList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(buyList[:, -(i + 1)], 0.9))
            badLegend = str(np.quantile(badList[:, -(i + 1)], 0.1)) + "," + str(np.quantile(badList[:, -(i + 1)], 0.5)) + "," + str(np.quantile(badList[:, -(i + 1)], 0.9))
            print(str(index) ,"_" , str(i) , "_" , colNameList[-(i+1)], "_" , buyLegend , " ", badLegend)
            plt.savefig('Plots/' + str(index) + "_" + str(i) + "_" + colNameList[-(i+1)] + '_box.pdf')
            plt.cla()
            plt.clf()
        for i in range(self.ngrams*4):
            a = {'Good': buyList[:, i],
                 'Bad': badList[:, i ]}
            df = pd.DataFrame.from_dict( a, orient='index')
            df = df.transpose()
            df.plot.box()
            #mustBuyLegend = str(np.quantile(mustBuyList[:, i ], 0.1)) + "," + str(np.quantile(mustBuyList[:, i ], 0.5)) + "," + str(np.quantile(mustBuyList[:, i ], 0.9))
            buyLegend = str(np.quantile(buyList[:, i], 0.1)) + "," + str(np.quantile(buyList[:, i], 0.5)) + "," + str(np.quantile(buyList[:, i], 0.9))
            badLegend = str(np.quantile(badList[:, i], 0.1)) + "," + str(np.quantile(badList[:, i], 0.5)) + "," + str(np.quantile(badList[:, i ], 0.9))
            print(str(index) ,"_" , str(i) , "_transac_", buyLegend , " ", badLegend)
            plt.savefig('Plots/' + str(index) + "_" + str(i) + "_transactions_box.pdf")
            plt.cla()
            plt.clf()
        plt.close()

    def __trimExtremes(self, listIn):
        # Score4 = -40, Score3 = -15 Score2 = -8 Score1 = -7 72 H = 0.4 48 H = 0.45 24 H = 0.55 6 H = 0.75
        data = np.array(listIn)
        data.reshape(-1, self.ngrams * 4 + ExtraFeatureCount + TransactionPeakHelper.PeakFeatureCount)
        tobeFilteredData = data[:, [-19, -17, -15, -13, -4, -3, -2, -1]]
        indexes = np.where((tobeFilteredData < [0.75, 0.55, 0.45, 0.4, -7.0, -8.0, -15.0, -40]).any(axis=1))
        return np.delete(listIn, indexes, 0)

    def __PrintImpl(self, inList, extraMessage):
        peakPatternValues = np.array(inList)
        print(extraMessage," len: ", len(peakPatternValues), " mean ", np.mean(peakPatternValues, axis=0))

    def __MergeInTransactions(self, transactionPeakHelper):
        # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
        for pattern in transactionPeakHelper.patternList:
            self.patternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.mustBuyList:
            self.mustBuyList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())

        for pattern in transactionPeakHelper.badPatternList:
            self.badPatternList.append(pattern.GetFeatures() + transactionPeakHelper.GetPeakFeatures())
