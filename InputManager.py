import Input as input
import numpy as np
import json
import TransactionHelper

class ReShapeManager:
    maxFeatureCount = 6
    minFeatureCount = 3

    def __init__( self ):
        self.inputs = []
        self.features = []
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            self.inputs.append( input.ReShapedInput(curBinCount) )
        self.scoreList = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.features = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.transactionHelper = TransactionHelper.TransactionAnalyzer()

    def addANewCurrency( self, jsonIn, transactionMSec, transactionCount, isAddOnlyTransactionPeaks ):
        peakData = jsonIn["peak"]
        riseAndTimeStrList = peakData.split(",")
        if len(riseAndTimeStrList) < 2:
            return

        transactionData = jsonIn["transactionList"]
        peakSize = len(riseAndTimeStrList)
        transactionDataSize = len(transactionData)
        assert peakSize == transactionDataSize
        riseAndTimeList = []
        for x in range(peakSize):
            riseMinute = input.RiseMinute(riseAndTimeStrList[x])
            riseAndTimeList.append(riseMinute)

        if not isAddOnlyTransactionPeaks:
            self.addLinePeaks(riseAndTimeList)
        self.transactionHelper.AddPeak( transactionData, riseAndTimeList,transactionMSec,transactionCount )


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
                    self.scoreList[curBinIndex][currentElemIndex] = self.__getScoreForButtomElement(elem, self.inputs[curBinIndex+1])
                elif elem[-1] > 0.0:
                    self.scoreList[curBinIndex][currentElemIndex] = self.__getScoreForRisingElement(elem, self.inputs[curBinIndex])

    def toFeaturesNumpy(self, binCount):
        curBinIndex = binCount - self.minFeatureCount
        return self.inputs[curBinIndex].toNumpy()

    def toResultsNumpy(self, binCount):
        curBinIndex = binCount - self.minFeatureCount
        newArray = list(map ( lambda elem: 0.0 if elem < 2.0 else 1.0, self.scoreList[curBinIndex] ))
        return np.array(newArray)

    def toTransactionFeaturesNumpy(self, transactionCount):
        return self.transactionHelper.toTransactionNumpy(transactionCount)

    def toTransactionResultsNumpy(self):
        return self.transactionHelper.toTransactionResultsNumpy()

    def resetScores(self):
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            #self.scoreList[curBinIndex].clear()
            self.scoreList[curBinIndex] = [0.0]*len(self.inputs[curBinIndex].inputRise);

    def __getScoreForButtomElement(self, oneSampleNBin, nPlusOneCompleteList):
        score = 2.0
        for elemList in nPlusOneCompleteList.inputRise:
            score+= self.__getScoreForButtom(oneSampleNBin, elemList)
        return score

    def __getScoreForRisingElement(self, oneSampleNBin, nBinCompleteList):
        score = -3.0
        for elemList in nBinCompleteList.inputRise :
            score+= self.__getScoreForRising(oneSampleNBin, elemList)
        return score

    def __checkPositivitySingleVal(self, lhs, rhs):
        if lhs * rhs < 0:
            return False
        diff = abs(lhs - rhs)
        if lhs < 5.0:
            return diff < 0.5
        elif lhs < 10.0:
            return diff < 0.75
        elif lhs < 15.0:
            return diff < 1.0
        else:
            return diff < 1.5

    def __getScoreForRising(self, oneSampleNBin, oneSampleOtherBin):
        isAllValid = all(
            [self.__checkPositivitySingleVal(x, y) for x, y in zip(oneSampleNBin[:-1], oneSampleOtherBin[:-1])])
        if not isAllValid:
            return 0
        if oneSampleNBin == oneSampleOtherBin:
            return 0
        #print( "Score will change ", *oneSampleNBin, " " , *oneSampleOtherBin)
        return max( -5.0, min(oneSampleOtherBin[-1] - oneSampleNBin[-1] - 2.5, 5.0) )

    def __getScoreForButtom(self, oneSampleNBin, oneSampleNPluseOneBin):
        isAllValid = all([self.__checkPositivitySingleVal(x, y) for x, y in zip(oneSampleNBin, oneSampleNPluseOneBin)])
        if  not isAllValid :
            return 0
        #print("Buttom repeat Score will change ", *oneSampleNBin, " ", *oneSampleNPluseOneBin)
        return min( 5.0, max( oneSampleNPluseOneBin[-1], -5.0 ) )
