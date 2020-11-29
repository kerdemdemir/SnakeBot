import numpy as np

import SuddenChangeTransactions
import DynamicTuner
import MarketStateManager as marketState
from datetime import datetime


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
        onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        for fileName in onlyfiles:
            self.__ReadFile( folderPath + "/" + fileName )

    def __ReadFile(self, fileName):
        with open(fileName) as fp:
            line = fp.readline()
            extraLineCount = 0
            if line.strip().split(",")[2] == "MarketState":
                extraLineCount = 1
                print("ExtraLine")
            line = fp.readline()
            while line:
                lineSplitList = line.strip().split(",")
                #print( "line: ", lineSplitList[0], "scoreList:  ", lineSplitList[1], "bid:  ", lineSplitList[11], "time:  ", lineSplitList[14])
                line = fp.readline()
                messageChangeTimeTransactionStrList = lineSplitList[0].split(";")
                priceStrList = messageChangeTimeTransactionStrList[1:9]
                timeStrList = messageChangeTimeTransactionStrList[9:17]
                transactionStrList = messageChangeTimeTransactionStrList[17:]

                resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
                resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
                resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

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
                    currentTransactionList = DynamicTuner.MergeTransactions(justTransactions, transParam.msec,
                                                                            transParam.gramCount)

                    datetime_object = datetime.strptime(lineSplitList[14+extraLineCount], '%Y-%b-%d %H:%M:%S')
                    epoch = datetime.utcfromtimestamp(0)
                    curSeconds = (datetime_object - epoch).total_seconds()
                    #marketStateList = self.marketState.getState(curSeconds)
                    #totalFeatures = currentTransactionList + marketStateList + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:] + priceStrList[-SuddenChangeTransactions.PeakFeatureCount:]
                    totalFeatures = currentTransactionList + resultsTimeFloat[-SuddenChangeTransactions.PeakFeatureCount:] + priceStrList[-SuddenChangeTransactions.PeakFeatureCount:]


                    self.totalLen = len(totalFeatures)
                    if float(lineSplitList[11+extraLineCount]) < 1.005:
                        self.totalFeaturesList[transactionIndex].append(totalFeatures)
                    else:
                        self.totalGoodFeaturesList[transactionIndex].append(totalFeatures)
                    #print(totalFeatures)
                    #print(len(totalFeatures), " ", totalFeatures)

