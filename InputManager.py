import Input as input
import numpy as np
import json
import TransactionHelper
import MarketStateManager


class ReShapeManager:
    maxFeatureCount = 8
    minFeatureCount = 3

    def __init__( self, transactionParams ):
        self.inputs = []
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            self.inputs.append( input.ReShapedInput(curBinCount) )
        self.transactionHelperList = []
        self.transactionParams = transactionParams
        self.marketState = MarketStateManager.MarketStateManager()
        for i in range(len(transactionParams)):
            self.transactionHelperList.append(TransactionHelper.TransactionAnalyzer(transactionParams[i].gramCount))

    def ClearMemory(self):
        print("Releasing memory")
        del self.transactionHelperList
        for input in self.inputs:
            del input.inputRise
            del input.inputTime

    def addNewFileData( self, jsonDictionary, isFinalize ):
        for jsonElem in jsonDictionary:
            self.addANewCurrency(jsonElem, isFinalize)

    def addANewCurrency( self, jsonIn, isFinalize ):
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
                                                       transParam.gramCount,self.maxFeatureCount, self.marketState if i == 0 else None, isFinalize )



    def addLinePeaks(self, riseAndTimeList):
        for curBinCount in range(self.minFeatureCount, self.maxFeatureCount):
            curBinIndex = curBinCount - self.minFeatureCount
            if len(riseAndTimeList) >= curBinCount:
                self.inputs[curBinIndex].concanate(riseAndTimeList)

    def finalize(self):
        counter = 0
        self.marketState.sort()
        for input in self.inputs:
            input.inputTime.clear()
            
        for transHelper in self.transactionHelperList:
            for peakHelper in transHelper.peakHelperList:
                peakHelper.SetMarketState( self.marketState.getState(peakHelper.peakTimeSeconds) )

        for transHelper in self.transactionHelperList:
            counter += 1
            transHelper.Finalize(counter)

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
        return self.transactionHelperList[index].toTransactionNumpy()

    def toTransactionResultsNumpy(self, index):
        return self.transactionHelperList[index].toTransactionResultsNumpy()

