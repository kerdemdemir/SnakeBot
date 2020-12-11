import copy
import bisect

PeakFeatureCount = 4
TransactionCountPerSecBase = 6
TransactionLimitPerSecBase = 0.3


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



def GetTotalPatternCount(ngrams):
    transCount = ngrams
    returnVal = 0
    for i in range(transCount // 2):
        returnVal += pow(2, i + 1)
    return returnVal

def ReduceToNGrams(listToMerge, ngrams):
    transCount = ngrams
    mergeRuleList = []
    returnVal = 0
    listToMerge.reverse()
    for i in range(transCount // 2):
        returnVal += pow(2, i + 1)
        mergeRuleList.append(returnVal)

    mergeListLen = len(listToMerge)
    curIndex = 2
    newMergeList = listToMerge[:2]
    while curIndex < mergeListLen:
        mergePos = bisect.bisect(mergeRuleList, curIndex)
        mergeSize = pow(2, mergePos)
        startData = listToMerge[curIndex]
        for k in range(mergeSize-1):
            if isinstance(startData, float):
                startData += listToMerge[curIndex+k+1]
            else:
                startData.CombineData(listToMerge[curIndex+k+1])
        curIndex += mergeSize
        newMergeList.append(startData)
    newMergeList.reverse()
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
        self.totalTransactionCount += 1
        if not isSell:
            self.transactionBuyCount += 1
            self.totalBuy += power
        else:
            self.totalSell += power

    def CombineData(self, otherData):
        self.totalTransactionCount += otherData.totalTransactionCount
        self.transactionBuyCount += otherData.transactionBuyCount
        self.totalBuy += otherData.totalBuy
        self.totalSell += otherData.totalSell
        self.lastPrice = otherData.lastPrice
        self.timeInSecs = otherData.timeInSecs

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
        self.priceDiff = 1.0
        self.priceMinRatio = 1.0
        self.marketStateList = []
        self.peaks = []
        self.timeList = []

    def SetPeaks(self, peakList, timeList):
        if PeakFeatureCount > 0:
            self.peaks = copy.deepcopy(peakList)
            self.timeList = copy.deepcopy(timeList)
            self.peaks[-1] += (self.priceDiff - 1.0)
            self.timeList[-1] += (self.timeDiffInSeconds//60)



    def Append(self, dataList, peakTime, jumpPrice, marketState):

        if len(dataList) > 0:
            priceList = list(map(lambda x: x.lastPrice, dataList))
            self.priceMaxRatio = dataList[-1].lastPrice / max(priceList)
            self.priceMinRatio = dataList[-1].lastPrice / min(priceList)
        lastTime = dataList[-1].timeInSecs
        if marketState:
            self.marketStateList = marketState.getState(lastTime)
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
        self.timeDiffInSeconds = lastTime - peakTime
        self.priceDiff = dataList[-1].lastPrice/jumpPrice


    def GetFeatures(self):
        returnList = []
        for i in range(len(self.transactionBuyList)):
            returnList.append(self.transactionBuyList[i])
            returnList.append(self.transactionSellList[i])
            returnList.append(self.transactionBuyPowerList[i])
            returnList.append(self.transactionSellPowerList[i])
        returnList.extend(self.marketStateList)
        if PeakFeatureCount > 0:
            returnList.extend(self.peaks[-PeakFeatureCount:])
            returnList.extend(self.timeList[-PeakFeatureCount:])

        return returnList

    def __repr__(self):
        return "list:%s,timeDiff:%d,totalBuy:%f,totalSell:%f,transactionCount:%f,score:%f" % (
            str(self.transactionBuyList), self.timeDiffInSeconds,
            self.totalBuy, self.totalSell,
            self.transactionCount, self.score)

