import Input as input
import numpy as np
import json
import TransactionHelper
import MarketStateManager

def keyMaker(list):
    key = 0
    for index in range(len(list)):
        key += int(min(999, int(abs(list[index]) * 10))* pow(1000, index))
    if list[0] < 0.0:
      key *= -1
    return key

class TransactionParam:
    def __init__ ( self, msec, gramCount ):
        self.msec = msec
        self.gramCount = gramCount

    def __repr__(self):
        return "MSec:%d,GramCount:%d" % (self.msec, self.gramCount)


class ReShapeManager:
    maxFeatureCount = 8
    minFeatureCount = 3

    def __init__( self, transactionParams ):
        self.inputs = []
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            self.inputs.append( input.ReShapedInput(curBinCount) )
        self.transactionHelperList = []
        self.transactionParams = transactionParams
        self.scoreMap = {}
        self.notMissCount = 0
        self.missCount = 0
        self.marketState = MarketStateManager.MarketStateManager()
        for _ in range(len(transactionParams)):
            self.transactionHelperList.append(TransactionHelper.TransactionAnalyzer())

    def ClearMemory(self):
        print("Releasing memory")
        del self.transactionHelperList
        for input in self.inputs:
            del input.inputRise
            del input.inputTime

    def addNewFileData( self, jsonDictionary ):
        for jsonElem in jsonDictionary:
            self.addANewCurrency(jsonElem)

    def addANewCurrency( self, jsonIn):
        peakData = jsonIn["peak"]
        riseAndTimeStrList = peakData.split(",")
        if len(riseAndTimeStrList) < 2:
            return

        transactionData = jsonIn["transactionList"]
        peakSize = len(riseAndTimeStrList)
        transactionDataSize = len(transactionData)
        if peakSize != transactionDataSize:
            print(peakSize , " ", transactionDataSize)
            return
        riseAndTimeList = []
        skipIndexes = []
        for x in range(peakSize):
            riseMinute = input.RiseMinute(riseAndTimeStrList[x])
            if len(riseAndTimeList) > 0 and riseAndTimeList[-1].rise * riseMinute.rise > 0:
                skipIndexes.append(x-1)
                riseAndTimeList[-1] = riseMinute
            else:
                riseAndTimeList.append(riseMinute)

        newTransParams = []
        for i in range(len(transactionData)):
            if i in skipIndexes:
                continue
            newTransParams.append(transactionData[i])

        transactionData = newTransParams

        startIndex = self.transactionHelperList[-1].GetStartIndex(transactionData)
        if startIndex == -1:
            return
        self.addLinePeaks(riseAndTimeList[startIndex:])

        for i in range(len(self.transactionParams)):
            transParam = self.transactionParams[i]
            self.transactionHelperList[i].AddCurrency( transactionData, riseAndTimeList,transParam.msec,
                                                       transParam.gramCount,self.maxFeatureCount, self.marketState if i == 0 else None )




    def addLinePeaks(self, riseAndTimeList):
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            if len(riseAndTimeList) >= curBinCount:
                self.inputs[curBinIndex].concanate(riseAndTimeList)

    def assignScores(self):
        counter = 0
        self.marketState.sort()
        for input in self.inputs:
            input.inputTime.clear()
            
        for transHelper in self.transactionHelperList:
            for peakHelper in transHelper.peakHelperList:
                peakHelper.SetMarketState( self.marketState.getState(peakHelper.peakTimeSeconds) )

        for transHelper in self.transactionHelperList:
            for peakHelper in transHelper.peakHelperList:
                scoreList = self.getScoreList(peakHelper.inputRise)
                peakHelper.scoreList = scoreList
            print("Finalizing index", counter, " Not missed " , self.notMissCount, " missed ", self.missCount)
            counter += 1
            transHelper.Finalize(counter)

    def getScore(self, list ):
        for curBinCount in range(self.maxFeatureCount-2, self.minFeatureCount-1, -1):
            curBinIndex = curBinCount - self.minFeatureCount
            lookUpList = list[-curBinCount:]

            score = 0.0
            curKey = keyMaker(lookUpList)
            if curKey in self.scoreMap:
                score = self.scoreMap[curKey]
            else:
                if lookUpList[-1] < 0.0:
                    score = self.__getScoreForButtomElement(lookUpList, self.inputs[curBinIndex+1].getSorter(), curBinIndex + 1)
                elif lookUpList[-1] > 0.0:
                    score = self.__getScoreForRisingElement(lookUpList, self.inputs[curBinIndex].getSorter(), curBinIndex + 1)
                self.scoreMap[curKey] = score

            if abs(score) > 6.0:
                return score

        return 0.0

    def getScoreList(self, list ):
        returnList = []
        for curBinCount in range(self.maxFeatureCount-2, self.minFeatureCount-1, -1):
            curBinIndex = curBinCount - self.minFeatureCount
            lookUpList = list[-curBinCount:]
            score = 0.0
            curKey = keyMaker(lookUpList)
            if curKey in self.scoreMap:
                score = self.scoreMap[curKey]
                returnList.append(score)
                self.notMissCount += 1
                continue
            self.missCount += 1
            if lookUpList[-1] < 0.0:
                score = self.__getScoreForButtomElement(lookUpList, self.inputs[curBinIndex+1].getSorter(), curBinIndex + 1)
            elif lookUpList[-1] > 0.0:
                score = self.__getScoreForRisingElement(lookUpList, self.inputs[curBinIndex].getSorter(), curBinIndex + 1)
            self.scoreMap[curKey] = score

            returnList.append(score)

        return returnList

    def setScore( self, transIndex, peakIndex, score ):
        self.transactionHelperList[transIndex].peakHelperList[peakIndex].score = score

    def toFeaturesNumpy(self, binCount):
        curBinIndex = binCount - self.minFeatureCount
        return self.inputs[curBinIndex].toNumpy()

    def toTransactionCurves(self, index):
        return self.transactionHelperList[index].toTransactionCurves()

    def toResultsNumpy(self, binCount):
        curBinIndex = binCount - self.minFeatureCount
        newArray = list(map ( lambda elem: 0.0 if elem < 6.0 else 1.0, self.scoreList[curBinIndex] ))
        return np.array(newArray)

    def toTestResultNumpy(self, xTest, gramCount):
        output = []
        for x in xTest:
            output.append( 1 if x[gramCount-1] < 0.0 else 0)
        return np.array(output)

    def toTransactionFeaturesNumpy(self, index ):
        return self.transactionHelperList[index].toTransactionNumpy(self.transactionParams[index].gramCount)

    def toTransactionResultsNumpy(self, index):
        return self.transactionHelperList[index].toTransactionResultsNumpy()

    def __getFactor(self, val, curIndex ):
        ngramFactor = lambda x :  0.8 + 0.15*x + 0.05*x*x
        if val < 4.0:
            return 0.5 * ngramFactor(curIndex)
        elif val < 5.0:
            return 0.6 * ngramFactor(curIndex)
        elif val < 7.5:
            return 0.8 * ngramFactor(curIndex)
        elif val < 10.0:
            return 1.0 * ngramFactor(curIndex)
        elif val < 15.0:
            return 1.5 * ngramFactor(curIndex)
        elif val < 20.0:
            return 2.0 * ngramFactor(curIndex)
        elif val < 30.0:
            return 3.0 * ngramFactor(curIndex)
        else:
            return 5.0 * ngramFactor(curIndex)

    def __getScoreForButtomElement(self, oneSampleNBin, nPlusOneCompleteInputSorter, curIndex):
        score = 0.0
        firstElem = oneSampleNBin[0]
        factor = self.__getFactor(abs(firstElem), curIndex)
        startIndex = nPlusOneCompleteInputSorter.getIndex( firstElem - factor)
        endIndex = nPlusOneCompleteInputSorter.getIndex( firstElem + factor)
        #print(firstElem, " ",curIndex , " ", startIndex, " ", endIndex, " ",  len(nPlusOneCompleteInputSorter.sortedPriceList), )
        for index in  range(startIndex, endIndex):
            elemList = nPlusOneCompleteInputSorter.sortedPriceList[index]
            score += self.__getScoreForButtom(oneSampleNBin, elemList, curIndex)
        return score

    def __getScoreForRisingElement(self, oneSampleNBin, nBinCompleteInputSorter, curIndex):
        score = 0
        firstElem = oneSampleNBin[0]
        factor = self.__getFactor(abs(firstElem), curIndex)
        startIndex = nBinCompleteInputSorter.getIndex( firstElem - factor)
        endIndex = nBinCompleteInputSorter.getIndex( firstElem + factor)

        for index in  range(startIndex, endIndex):
            elemList = nBinCompleteInputSorter.sortedPriceList[index]
            score += self.__getScoreForRising(oneSampleNBin, elemList, curIndex)
        return score


    def __checkPositivitySingleVal(self, lhs, rhs, curIndex):
        ngramFactor = lambda x :  0.8 + 0.15*x + 0.05*x*x
        if lhs * rhs < 0:
            return False
        diff = abs(lhs - rhs)
        lshAbs = abs(lhs)
        if lshAbs < 4.0:
            return diff < 0.5 * ngramFactor(curIndex)
        elif lshAbs < 5.0:
            return diff < 0.6 * ngramFactor(curIndex)
        elif lshAbs < 7.5:
            return diff < 0.8 * ngramFactor(curIndex)
        elif lshAbs < 10.0:
            return diff < 1.0 * ngramFactor(curIndex)
        elif lshAbs < 15.0:
            return diff < 1.5 * ngramFactor(curIndex)
        elif lshAbs < 20.0:
            return diff < 2.0 * ngramFactor(curIndex)
        elif lshAbs < 30.0:
            return diff < 3.0 * ngramFactor(curIndex)
        else:
            return diff < 5.0 * ngramFactor(curIndex)

    def __getScoreForRising(self, oneSampleNBin, oneSampleOtherBin, curIndex):

        if not self.__checkPositivitySingleVal(oneSampleNBin[0],oneSampleOtherBin[0], curIndex):
            return False
        isAllValid = all(
            [self.__checkPositivitySingleVal(x, y, curIndex) for x, y in zip(oneSampleNBin[:-1], oneSampleOtherBin[:-1])])
        if not isAllValid:
            return 0.0
        diff = oneSampleOtherBin[-1] - oneSampleNBin[-1]
        if diff < -1.5:
            return 0.0

        return self.__clampVal( diff - 4.0 )

    def __getScoreForButtom(self, oneSampleNBin, oneSampleNPluseOneBin, curIndex):
        if not self.__checkPositivitySingleVal(oneSampleNBin[0],oneSampleNPluseOneBin[0], curIndex):
            return False
        isAllValid = all([self.__checkPositivitySingleVal(x, y, curIndex) for x, y in zip(oneSampleNBin[:-1], oneSampleNPluseOneBin[:-2])])
        if not isAllValid:
            return 0.0
        diff = oneSampleNBin[-1] - oneSampleNPluseOneBin[-2]
        if diff < -1.5:
            return 0.0
        elif diff < 1.5:
            return self.__clampVal(oneSampleNPluseOneBin[-1] - abs(diff))#Positive effect
        else:
            return -self.__clampVal(diff)#Negative effect

    def __clampVal(self, val ):
        return min(5.0, max(val, -5.0))
