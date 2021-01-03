import numpy as np

import SuddenChangeTransactions
import TransactionBasics
import DynamicTuner
import MarketStateManager as marketState
from datetime import datetime
import bisect


from os import listdir
from os.path import isfile, join

class ExtraDataManager:
    def __init__( self, readFolderPath, transParamListIn, marketStateParam ):
        self.transParamList = transParamListIn
        self.marketState = marketStateParam
        self.totalFeaturesList = []
        self.totalGoodFeaturesList = []
        for i in range(len(transParamListIn)):
            self.totalFeaturesList.append([])
            self.totalGoodFeaturesList.append([])
        self.totalLen = 0
        self.goodCount = 0
        self.badCount = 0
        self.strList = []
        self.ReadFiles( readFolderPath )

    def getNumpy(self,index):
        data = np.array(self.totalFeaturesList[index]+self.totalGoodFeaturesList[index])
        data.reshape(-1, self.totalLen)
        return data

    def concanate(self, dataIn, index ):
        data = np.array(self.totalFeaturesList[index])
        data.reshape(-1, self.totalLen)
        return np.concatenate((dataIn, data), axis=0)

    def getResult(self,index):
        return [0] * len(self.totalFeaturesList[index])

    def getConcanatedResult(self,index):
        return [0] * len(self.totalFeaturesList[index]) + [1] * len(self.totalGoodFeaturesList[index])

    def ReadFiles(self, folderPath ):
        self.goodCount = 0
        self.badCount = 0
        onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        for fileName in onlyfiles:
            self.__ReadFile( folderPath + "/" + fileName )
        print("Good count: ", self.goodCount , " Bad count: ", self.badCount)

    def __ReadFile(self, fileName):
        with open(fileName) as fp:
            line = fp.readline()
            extraLineCount = 0

            if "lock" in fileName:
                return

            if line.strip().split(",")[2] == "MarketState":
                extraLineCount = 1

            if line.strip().split(",")[6] == "SellResults":
                extraLineCount += 1
                print("ExtraLine")

            if line.strip().split(",")[7] == "InitialSellResult":
                extraLineCount += 1
                print("ExtraLine")

            if line.strip().split(",")[4] == "BargainRatio":
                extraLineCount += 2
                print("ExtraLine")
            print(fileName, " ", extraLineCount)
            line = fp.readline()
            while line:
                lineSplitList = line.strip().split(",")
                #print( "line: ", lineSplitList[0], "scoreList:  ", lineSplitList[1], "bid:  ", lineSplitList[11], "time:  ", lineSplitList[14])
                line = fp.readline()
                messageChangeTimeTransactionStrList = lineSplitList[0].split(";")
                priceStrList = messageChangeTimeTransactionStrList[1:9]
                timeStrList = messageChangeTimeTransactionStrList[9:17]
                transactionStrList = messageChangeTimeTransactionStrList[17:-8]
                #print(transactionStrList)
                resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
                resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
                resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]
                TransactionBasics.RiseListSanitizer(resultsChangeFloat, resultsTimeFloat)
                if float(resultsChangeFloat[-1]) > 1.0:
                    continue
                totalPower = resultsTransactionFloat[-1]+resultsTransactionFloat[-2]
                totalTransCount = resultsTransactionFloat[-3] + resultsTransactionFloat[-4]
                if totalPower < TransactionBasics.TransactionLimitPerSecBase or totalTransCount < TransactionBasics.TransactionCountPerSecBase:
                    continue
                if float(lineSplitList[8+extraLineCount]) > 0.01:
                    continue
                resultStr = ""
                #scores = list(map(lambda x: float(x), lineSplitList[1][1:-1].split(";")))
                #print("scores ", scores )
                # totalPerExtra = transHelper.ExtraPerDataInfo * len(transParamList)
                #minMaxPriceRatio = SuddenChangeTransactions.GetPeaksRatio(resultsChangeFloat, 7)
                #print(resultsChangeFloat, " ", resultsChangeFloat[7], " ", minMaxPriceRatio)

                for transactionIndex in range(len(self.transParamList)):
                    transParam = self.transParamList[transactionIndex]

                    justTransactions = resultsTransactionFloat
                    #if len(justTransactions) != 80:
                        #print("Bad extra trans data ", len(justTransactions))
                        #continue
                    #print( len(justTransactions), " ", justTransactions)
                    multipliedGramCount = TransactionBasics.GetTotalPatternCount(transParam.gramCount)
                    currentTransactionList = DynamicTuner.MergeTransactions(justTransactions, transParam.msec, multipliedGramCount)
                    if len(currentTransactionList) == 0:
                        continue

                    basicList = TransactionBasics.CreateTransactionList(currentTransactionList)
                    basicList = TransactionBasics.ReduceToNGrams(basicList, transParam.gramCount)
                    currentTransactionList = TransactionBasics.GetListFromBasicTransData(basicList)
                    datetime_object = datetime.strptime(lineSplitList[14+extraLineCount], '%Y-%b-%d %H:%M:%S')
                    epoch = datetime.utcfromtimestamp(0)
                    curSeconds = (datetime_object - epoch).total_seconds()

                    buyPrice = float(lineSplitList[13 + extraLineCount])
                    #maxMinDataList2 = TransactionBasics.GetMaxMinListWithTime(lineSplitList[20 + extraLineCount], curSeconds, buyPrice)
                    maxMinStrList = lineSplitList[1].split(";")
                    maxMinDataList = [float(messageStr) for messageStr in maxMinStrList]
                    if not TransactionBasics.IsUseMaxInList:
                        maxMinDataList = [maxMinDataList[0],maxMinDataList[2],maxMinDataList[4],maxMinDataList[6]]
                    #if maxMinDataList[1] < 0.95:
                        #print("Alert3 ", maxMinDataList[0], " ", float(lineSplitList[15 + extraLineCount]))
                        #continue
                    if self.marketState:
                        marketStateList = self.marketState.getState(curSeconds)
                    else:
                        marketStateList = []


                    #totalFeatures = currentTransactionList + marketStateList + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:] + priceStrList[-SuddenChangeTransactions.PeakFeatureCount:]
                    if SuddenChangeTransactions.PeakFeatureCount > 0:
                        totalFeatures = currentTransactionList + marketStateList + resultsChangeFloat[-SuddenChangeTransactions.PeakFeatureCount:] \
                                        + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:] + maxMinDataList
                        #totalFeatures = currentTransactionList + priceStrList[-SuddenChangeTransactions.PeakFeatureCount:] + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:]
                        #totalFeatures = currentTransactionList + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:]
                    else :
                        #totalFeatures = currentTransactionList + marketStateList
                        totalFeatures = currentTransactionList + marketStateList + maxMinDataList

                    self.totalLen = len(totalFeatures)
                    if float(lineSplitList[11+extraLineCount]) < 1.0:
                        self.totalFeaturesList[transactionIndex].append(totalFeatures)
                        self.badCount+=1
                    elif float(lineSplitList[11+extraLineCount]) > 1.002:
                        self.totalGoodFeaturesList[transactionIndex].append(totalFeatures)
                        self.goodCount+=1

                    self.strList.append(lineSplitList[0])
                    #print(totalFeatures)
                    #print(len(totalFeatures), " ", totalFeatures)
