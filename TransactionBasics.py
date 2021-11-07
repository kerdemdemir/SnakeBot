import copy
import bisect
from datetime import datetime
import bisect
import statistics

PeakFeatureCount = 2
MaximumSampleSizeFromPattern = 100
MaximumSampleSizeFromGoodPattern = 1
TransactionCountPerSecBase = 3
TransactionLimitPerSecBase = 0.1
TotalPowerLimit = 0.5
TotalElementLimitMsecs = 10000
MaxMinListTimes = [60*60*6,60*60*24,60*60*36]
IsUseMaxInList = True
MergeLenghtSize = 2

def GetMaxMinList(maxMinList):
    extraCount = len(MaxMinListTimes)
    if extraCount == 0:
        return []

    returnVal = []
    for index in range(extraCount * 2):
        if index % 2 == 0:
            returnVal.append(maxMinList[index])
        elif IsUseMaxInList:
            returnVal.append(maxMinList[index])
    return returnVal

def LastNElementsTransactionPower(list, index, elementCount):
    totalTradePower = 0
    for curIndex in range(index-elementCount, index+1):
        lastTotalTradePower = list[curIndex].totalBuy + list[curIndex].totalSell
        totalTradePower += lastTotalTradePower
    return totalTradePower

class TimePriceBasic:
    def __init__( self, timeInSeconds, priceIn ) :
        self.timeInSec = timeInSeconds
        self.price = priceIn

    def __lt__(self, other):
        return self.timeInSec < other.timeInSec


def GetMaxMinListWithTime(allPeaksStr, buyTimeInSeconds, buyPrice):
    activePeak = allPeaksStr.split("|")[0]
    allPeakListStr = activePeak.split('&')
    allPeakList = []
    epoch = datetime.utcfromtimestamp(0)
    for peakStr in allPeakListStr:
        price = float(peakStr.split(" ")[0])
        timeStr = peakStr.split(" ")[1]
        datetime_object = datetime.strptime(timeStr, '%Y%m%dT%H%M%S')
        curSeconds = (datetime_object - epoch).total_seconds()
        if curSeconds > buyTimeInSeconds:
            break
        curTimePrice = TimePriceBasic(curSeconds, price)
        allPeakList.append(curTimePrice)

    returnValue = []
    for cureTimeOffset in MaxMinListTimes:
        curTimeInSeconds = buyTimeInSeconds - cureTimeOffset
        startIndex = bisect.bisect_right(allPeakList, TimePriceBasic(curTimeInSeconds,0.0))
        # if len(allPeakList) > startIndex:
        #     timeTemp = allPeakList[startIndex].timeInSec
        #     print("Alert " , timeTemp-curTimeInSeconds, " ", timeTemp," " ,curTimeInSeconds, " ", buyTimeInSeconds)
        # else:
        #     print("Alert2 ", len(allPeakList), " ", startIndex)
        curMax = 1.0
        curMin = 1.0
        for peak in allPeakList[startIndex:]:
            curMin = min( peak.price/buyPrice, curMin )
            curMax = max( peak.price/buyPrice, curMax )
        returnValue.append(curMin)
        if IsUseMaxInList:
            returnValue.append(curMax)
    return returnValue

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def CreateTransactionList(inList):
    transPatternReturnVal = []
    newList = list(chunks(inList, 4))
    for listElem in newList:
        elem = BasicTransactionData(listElem)
        transPatternReturnVal.append(elem)
    return transPatternReturnVal

def GetListFromBasicTransData( inBasicTransactionDataList ):
    returnList = []
    for i in range(len(inBasicTransactionDataList)):
        returnList.append(inBasicTransactionDataList[i].transactionBuyCount)
        returnList.append(inBasicTransactionDataList[i].transactionSellCount)
        returnList.append(inBasicTransactionDataList[i].totalBuy)
        returnList.append(inBasicTransactionDataList[i].totalSell)
    return returnList

def RiseListSanitizer( riseList, timeList ):
    correctIndexes = []
    for i in range(len(riseList) - 1):
        if riseList[i] * riseList[i + 1] > 0.0:
            correctIndexes.append( i+1 )

    for index in correctIndexes:
        timeValue = timeList[index - 1]
        timeList[index - 1] = timeValue//2
        timeList.insert(index,timeValue//2)
        if riseList[index - 1] > 0.0 :
            riseList.insert(index, -3.0)
        else:
            riseList.insert(index, 3.0)

def GetTotalPatternCount(ngrams):
    return 361

def ReduceToNGrams(listToMerge, ngrams):
    elemList = [330, 30, 1]
    startIndex = 0
    newMergeList = []
    for mergeSize in elemList:
        startData = listToMerge[startIndex]
        for k in range(mergeSize-1):
            startData.CombineData(listToMerge[startIndex+k+1])
        startIndex += mergeSize
        startData.Divide(mergeSize)
        newMergeList.append(startData)
    return newMergeList

class TransactionParam:
    def __init__ ( self, msec, gramCount ):
        self.msec = msec
        self.gramCount = gramCount

    def __repr__(self):
        return "MSec:%d,GramCount:%d" % (self.msec, self.gramCount)

class BasicTransactionData:
    def __init__(self, list):
        self.totalBuy = list[2]
        self.totalSell = list[3]
        self.transactionBuyCount = list[0]
        self.transactionSellCount = list[1]

    def CombineData(self, otherData):
        self.transactionSellCount += otherData.transactionSellCount
        self.transactionBuyCount += otherData.transactionBuyCount
        self.totalBuy += otherData.totalBuy
        self.totalSell += otherData.totalSell

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
        self.maxPrice = 0.0
        self.minPrice = 1000.0
        self.startIndex = 0
        self.endIndex = 0

    def __repr__(self):
        return "TotalBuy:%f,TotalSell:%f,TransactionCount:%f,Score:%f,LastPrice:%f,Time:%d" % (
        self.totalBuy, self.totalSell,
        self.transactionBuyCount, self.score, self.lastPrice, self.timeInSecs)

    # "m": true, "l": 6484065,"M": true,"q": "44113.00000000","a": 5378484,"T": 1591976004949,"p": "0.00000225","f": 6484064
    def AddData(self, jsonIn):
        isSell = bool(jsonIn["m"])
        power = float(jsonIn["q"]) * float(jsonIn["p"])
        if self.firstPrice == 0.0:
            self.firstPrice = float(jsonIn["p"])
        self.lastPrice = float(jsonIn["p"])
        self.maxPrice = max(self.lastPrice, self.maxPrice)
        self.minPrice = min(self.lastPrice, self.minPrice)
        self.totalTransactionCount += 1
        if not isSell:
            self.transactionBuyCount += 1
            self.totalBuy += power
        else:
            self.totalSell += power
            
    def Divide(self, dividor):
        self.transactionBuyCount /= dividor
        self.totalTransactionCount /= dividor
        self.totalSell /= dividor
        self.totalBuy /= dividor

    def CombineData(self, otherData):
        self.totalTransactionCount += otherData.totalTransactionCount
        self.transactionBuyCount += otherData.transactionBuyCount
        self.totalBuy += otherData.totalBuy
        self.totalSell += otherData.totalSell
        self.lastPrice = otherData.lastPrice
        if self.firstPrice == 0.0:
            self.firstPrice = otherData.firstPrice
        self.timeInSecs = otherData.timeInSecs
        self.endIndex = otherData.endIndex
        self.timeInSecs = otherData.timeInSecs
        self.maxPrice = max(self.maxPrice, otherData.maxPrice)
        if otherData.minPrice != 0.0:
            self.minPrice = min(self.minPrice, otherData.minPrice)

    def SetTime(self, timeInSecs):
        self.timeInSecs = timeInSecs

    def SetIndex(self, index):
        if self.startIndex == 0:
            self.startIndex = index
        self.endIndex = index

    def Reset(self):
        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionBuyCount = 0.0
        self.totalTransactionCount = 0.0
        self.score = 0
        self.timeInSecs = 0
        self.firstPrice = 0.0
        self.lastPrice = 0.0
        self.maxPrice = 0.0
        self.minPrice = 1000.0
        self.startIndex = 0
        self.endIndex = 0

class TransactionPattern:
    def __init__(self):
        self.transactionBuyList = []
        self.transactionSellList = []
        self.transactionBuyPowerList = []
        self.transactionSellPowerList = []
        self.firstLastPriceList = []
        self.minMaxPriceList = []
        self.upDownRangeBuyRatio = 10000000.0
        self.upDownRangeSellRatio = 10000000.0
        self.upDownRangeBuyRatioCount = 10000000.0
        self.upDownRangeSellRatioCount = 10000000.0

        self.detailedTransactionList = []
        self.detailedHighestPowerNumber = 0.0
        self.detailedHighestCountNumber = 0
        self.detailLen = 0.0
        self.detailedHighestSellCountNumber = 0.0
        self.isMaxBuyLater = False

        self.maxDetailBuyCount = 0.0
        self.maxDetailBuyPower = 0.0

        self.lastDownRatio = 0.0
        self.lastUpRatio = 0.0
        self.lastRatio = 0.0

        self.upDownRangeRatio = 0.0

        self.totalBuy = 0.0
        self.totalSell = 0.0
        self.transactionCount = 0.0
        self.score = 0.0
        self.totalTransactionCount = 0
        self.timeDiffInSeconds = 0
        self.priceMaxRatio = 1.0
        self.priceDiff = 1.0
        self.priceMinRatio = 1.0
        self.marketStateList = []
        self.peaks = []
        self.timeList = []
        self.isAvoidPeaks = True
        self.buyRatio = 1.0
        self.buyTimeDiffInSecs = 0
        self.buyInfoEnabled  = False

        self.totalPeakCount15M = 0
        self.totalPeakCount1Hour = 0
        self.totalPeakCount6Hour = 0
        self.totalPeakCount24Hour = 0


    def SetDetailedTransaction(self, detailedTransactionList, dataRange):
        self.detailedTransactionList = detailedTransactionList

        buyPowerList = list(map(lambda x: x.totalBuy, detailedTransactionList))
        buyCountList = list(map(lambda x: x.transactionBuyCount, detailedTransactionList))
        buySellList = list(map(lambda x: x.totalSell, detailedTransactionList))
        self.detailLen = len(buyPowerList)

        self.maxDetailBuyCount = max(buyCountList)
        self.maxDetailBuyPower = max(buyPowerList)
        maxSellVal = max(buySellList)
        if maxSellVal < 0.05:
            self.isMaxBuyLater = True
        else:
            self.isMaxBuyLater = buyPowerList.index(max(buyPowerList)) >= buySellList.index(maxSellVal)

    def SetPeaks(self, peakList, timeList, ratio, timeDif ):
        if PeakFeatureCount > 0:
            self.peaks = copy.deepcopy(peakList)
            self.timeList = copy.deepcopy(timeList)

        self.peaks[-1] += ratio
        self.timeList[-1] += timeDif

        totalTime = 0
        for curTime in reversed(self.timeList):
            totalTime += curTime
            if totalTime < 15 :
                self.totalPeakCount15M += 1
                self.totalPeakCount1Hour += 1
                self.totalPeakCount6Hour += 1
                self.totalPeakCount24Hour += 1
            elif totalTime < 60 :
                self.totalPeakCount1Hour += 1
                self.totalPeakCount6Hour += 1
                self.totalPeakCount24Hour += 1
            elif totalTime < 360:
                self.totalPeakCount6Hour += 1
                self.totalPeakCount24Hour += 1
            elif totalTime < 2160:
                self.totalPeakCount24Hour += 1
            else:
                break

        if self.peaks[-1] < 0.0:
            self.lastDownRatio = self.peaks[-1]+self.peaks[-2]
            self.lastUpRatio = self.peaks[-2] + self.peaks[-3]
        else:
            self.lastDownRatio = self.peaks[-2]+self.peaks[-3]
            self.lastUpRatio = self.peaks[-1] + self.peaks[-2]

        self.lastRatio = self.peaks[-1]
        self.isAvoidPeaks = False

    def Append(self, dataList, peakTime, jumpPrice, marketState):

        minList = list(map(lambda x: x.minPrice, dataList))
        minOfMin = min(minList)

        if len(dataList) > 0:
            priceList = list(map(lambda x: x.lastPrice, dataList))
            self.priceMaxRatio = dataList[-1].lastPrice / max(priceList)
            self.priceMinRatio = dataList[-1].lastPrice / min(priceList)

        self.priceMinRatio = minOfMin / dataList[-1].lastPrice
        lastTime = dataList[-1].timeInSecs
        if marketState:
            self.marketStateList = marketState.getState(lastTime)
        else:
            self.marketStateList = []

        if dataList[0].totalBuy != 0.0:
            self.upDownRangeBuyRatio = dataList[1].totalBuy/dataList[0].totalBuy
        if dataList[0].totalSell != 0.0:
            self.upDownRangeSellRatio = dataList[1].totalSell/dataList[0].totalSell
        if dataList[0].transactionBuyCount != 0.0:
            self.upDownRangeBuyRatioCount = dataList[1].transactionBuyCount/dataList[0].transactionBuyCount




        for elem in dataList:
            self.transactionBuyList.append(elem.transactionBuyCount)
            self.transactionSellList.append(elem.totalTransactionCount - elem.transactionBuyCount)
            self.transactionBuyPowerList.append(elem.totalBuy)
            self.transactionSellPowerList.append(elem.totalSell)
            self.minMaxPriceList.append(elem.maxPrice/elem.minPrice)
            self.firstLastPriceList.append(elem.lastPrice/elem.firstPrice)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionBuyCount
            self.totalTransactionCount += elem.totalTransactionCount
            self.timeDiffInSeconds = lastTime - peakTime
            self.priceDiff = dataList[-1].lastPrice/jumpPrice


    def AppendWithOutPeaks(self, dataList, marketState, buyPrice, buyTimeInSecs):

        lastTime = dataList[-1].timeInSecs
        self.buyTimeDiffInSecs = lastTime - buyTimeInSecs
        self.buyRatio = dataList[-1].lastPrice/buyPrice
        self.buyInfoEnabled = False
        if marketState:
            self.marketStateList = marketState.getState(lastTime)[0:2]
            self.marketStateList.extend(marketState.getState(buyTimeInSecs)[0:2])
        else:
            self.marketStateList = []
        for elem in dataList:
            self.transactionBuyList.append(elem.transactionBuyCount)
            self.transactionSellList.append(elem.totalTransactionCount - elem.transactionBuyCount)
            self.transactionBuyPowerList.append(elem.totalBuy)
            self.transactionSellPowerList.append(elem.totalSell)
            self.totalBuy += elem.totalBuy
            self.totalSell += elem.totalSell
            self.transactionCount += elem.transactionBuyCount
            self.totalTransactionCount += elem.totalTransactionCount


    def GetFeatures(self):
        returnList = []
        returnList.append(self.upDownRangeBuyRatio)
        returnList.append(self.upDownRangeSellRatio)
        returnList.append(self.upDownRangeBuyRatioCount)
        for i in range(len(self.transactionBuyList)):
            returnList.append(self.transactionBuyList[i])
            returnList.append(self.transactionSellList[i])
            returnList.append(self.transactionBuyPowerList[i])
            returnList.append(self.transactionSellPowerList[i])
            returnList.append(self.minMaxPriceList[i])
            returnList.append(self.firstLastPriceList[i])
        returnList.append(self.detailLen)
        returnList.append(self.maxDetailBuyCount)
        returnList.append(self.maxDetailBuyPower)
        returnList.append(self.detailedHighestSellCountNumber)
        returnList.append(self.priceMinRatio)

        returnList.extend(self.marketStateList)
        if PeakFeatureCount > 0 and not self.isAvoidPeaks:
            returnList.extend(self.peaks[-PeakFeatureCount:])
            returnList.extend(self.timeList[-PeakFeatureCount:])

        returnList.append(self.lastDownRatio)
        returnList.append(self.lastUpRatio)
        if self.buyInfoEnabled:
            returnList.append(self.buyRatio)
            returnList.append(self.buyTimeDiffInSecs)

        returnList.append(self.totalPeakCount15M)
        returnList.append(self.totalPeakCount1Hour)
        returnList.append(self.totalPeakCount6Hour)
        returnList.append(self.totalPeakCount24Hour)

        return returnList

    def __repr__(self):
        return "list:%s,timeDiff:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f" % (
            str(self.transactionBuyList), self.timeDiffInSeconds,
            self.totalBuy, self.totalSell,
            self.transactionCount, self.score)

