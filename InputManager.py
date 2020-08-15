import Input as input
import numpy as np
import json
import TransactionHelper

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
        self.features = []
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            self.inputs.append( input.ReShapedInput(curBinCount) )
        self.scoreList = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.features = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.transactionHelperList = []
        self.transactionParams = transactionParams
        for _ in range(len(transactionParams)):
            self.transactionHelperList.append(TransactionHelper.TransactionAnalyzer())

    def addANewCurrency( self, jsonIn, isAddOnlyTransactionPeaks ):
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
        for x in range(peakSize):
            riseMinute = input.RiseMinute(riseAndTimeStrList[x])
            riseAndTimeList.append(riseMinute)

        if not isAddOnlyTransactionPeaks:
            self.addLinePeaks(riseAndTimeList)

        for i in range(len(self.transactionParams)):
            transParam = self.transactionParams[i]
            self.transactionHelperList[i].AddPeak( transactionData, riseAndTimeList,transParam.msec,transParam.gramCount,self.maxFeatureCount )


    def addLinePeaks(self, riseAndTimeList):
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            if len(riseAndTimeList) >= curBinCount:
                self.inputs[curBinIndex].concanate(riseAndTimeList)



    def assignScores(self):
        self.resetScores()
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            for currentElemIndex in range(len(self.inputs[curBinIndex].inputRise)):
                elem = self.inputs[curBinIndex].inputRise[currentElemIndex]
                if elem[-1] < 0.0 and curBinIndex + 1 < self.maxFeatureCount - self.minFeatureCount:
                    self.scoreList[curBinIndex][currentElemIndex] = self.__getScoreForButtomElement(elem, self.inputs[curBinIndex+1].inputRise, curBinIndex+1)
                elif elem[-1] > 0.0:
                    self.scoreList[curBinIndex][currentElemIndex] = self.__getScoreForRisingElement(elem, self.inputs[curBinIndex].inputRise, curBinIndex+1)

    def toFeaturesNumpy(self, binCount):
        curBinIndex = binCount - self.minFeatureCount
        return self.inputs[curBinIndex].toNumpy()

    def toTransactionCurvesToNumpy(self, index , binCount):
        return self.transactionHelperList[index].toTransactionCurvesToNumpy(binCount)

    def toTransactionScores(self, index):
        self.resetScores()
        curTransactionHelper = self.transactionHelperList[index]
        print(" Transaction scores ", len(curTransactionHelper.peakHelperList))
        results = [ 0.0 ] * len(curTransactionHelper.peakHelperList)
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            curBins = curTransactionHelper.toTransactionCurves(curBinCount)
            curBinsPlusOne = curTransactionHelper.toTransactionCurves(curBinCount+1)
            print( len(curBins))
            for currentElemIndex in range(len(curBins)):
                elem = curBins[currentElemIndex]
                if elem[-1] < 0.0 and curBinIndex + 1 < (self.maxFeatureCount - self.minFeatureCount):
                    results[currentElemIndex] += self.__getScoreForButtomElement(elem, curBinsPlusOne, curBinIndex+1)
                elif elem[-1] > 0.0:
                    results[currentElemIndex] += self.__getScoreForRisingElement(elem, curBins, curBinIndex+1)

        results = list(map(lambda x: 0.0 if x/(self.maxFeatureCount - self.minFeatureCount) < 4.0 else 1.0, results ))
        return np.array(results)


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

    def toTransactionPeakResultsNumpy(self, index):
        return self.transactionHelperList[index].toTransactionPeakResultsNumpy()


    def resetScores(self):
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            #self.scoreList[curBinIndex].clear()
            self.scoreList[curBinIndex] = [0.0]*len(self.inputs[curBinIndex].inputRise)

    def __getScoreForButtomElement(self, oneSampleNBin, nPlusOneCompleteList, curIndex):
        score = 0.0
        for elemList in  nPlusOneCompleteList:
            score+= self.__getScoreForButtom(oneSampleNBin, elemList, curIndex)
        return score

    def __getScoreForRisingElement(self, oneSampleNBin, nBinCompleteList, curIndex):
        score = 0
        for elemList in nBinCompleteList :
            score+= self.__getScoreForRising(oneSampleNBin, elemList, curIndex)
        return score


    def __checkPositivitySingleVal(self, lhs, rhs, curIndex):
        ngramFactor = lambda x :  0.8 + 0.15*x + 0.05*x*x
        if lhs * rhs < 0:
            return False
        diff = abs(lhs - rhs)
        if lhs < 5.0:
            return diff < 0.5 * ngramFactor(curIndex)
        elif lhs < 10.0:
            return diff < 0.75 * ngramFactor(curIndex)
        elif lhs < 15.0:
            return diff < 1.0 * ngramFactor(curIndex)
        else:
            return diff < 1.5 * ngramFactor(curIndex)

    def __getScoreForRising(self, oneSampleNBin, oneSampleOtherBin, curIndex):

        if not self.__checkPositivitySingleVal(oneSampleNBin[0],oneSampleOtherBin[0], curIndex):
            return False
        isAllValid = all(
            [self.__checkPositivitySingleVal(x, y, curIndex) for x, y in zip(oneSampleNBin[:-1], oneSampleOtherBin[:-1])])
        if not isAllValid:
            return 0
        diff = oneSampleOtherBin[-1] - oneSampleNBin[-1]
        if diff < -1:
            return 0

        return self.__clampVal( diff - 4.0 )

    def __getScoreForButtom(self, oneSampleNBin, oneSampleNPluseOneBin, curIndex):
        if not self.__checkPositivitySingleVal(oneSampleNBin[0],oneSampleNPluseOneBin[0], curIndex):
            return False
        isAllValid = all([self.__checkPositivitySingleVal(x, y, curIndex) for x, y in zip(oneSampleNBin[:-1], oneSampleNPluseOneBin[:-2])])
        if not isAllValid:
            return 0
        diff = oneSampleNBin[-1] - oneSampleNPluseOneBin[-2]
        if diff < -1:
            return 0
        elif diff < 1:
            return self.__clampVal(oneSampleNPluseOneBin[-1] - abs(diff))#Positive effect
        else:
            return -self.__clampVal(diff)#Negative effect

    def __clampVal(self, val ):
        return min(5.0, max(val, -5.0))