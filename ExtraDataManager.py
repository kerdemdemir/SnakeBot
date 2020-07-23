import numpy as np


from os import listdir
from os.path import isfile, join

class ExtraDataManager:
    def __init__( self, minBin, maxBin, transCount, readFolderPath ):
        self.maxFeatureCount = maxBin
        self.minFeatureCount = minBin
        self.transCount = transCount
        self.scoreList = []
        self.featureArr  = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.featureArrNumpy = [[] for _ in range(self.maxFeatureCount - self.minFeatureCount)]
        self.transactionFeaturesList = []
        self.transactionFeaturesListNumpy = []
        self.ReadFiles( readFolderPath )

    def ConcanateFeature( self, currentNumpyArray, binCount ):
        binIndex = binCount-self.minFeatureCount
        self.featureArrNumpy[binIndex] = np.array(self.featureArr[binIndex])
        self.featureArrNumpy[binIndex].reshape(-1, binCount*2)
        return np.concatenate((currentNumpyArray, self.featureArrNumpy[binIndex]))

    def ConcanateResults ( self, currentNumpyArray ):
        return np.concatenate((currentNumpyArray, np.array(self.scoreList)))

    def ConcanateTransactions ( self, currentNumpyArray, transactionCount ):
        self.transactionFeaturesListNumpy = np.array(self.transactionFeaturesList)
        self.transactionFeaturesListNumpy.reshape(-1, transactionCount)
        return np.concatenate((currentNumpyArray, self.transactionFeaturesListNumpy))

    def ReadFiles(self, folderPath ):
        onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        for fileName in onlyfiles:
            self.__ReadFile( folderPath + "/" + fileName )

    def __ReadFile(self, fileName):
        with open(fileName) as fp:
            line = fp.readline()
            line = fp.readline()
            while line:
                lineSplitList = line.strip().split(",")
                print( "line: ", lineSplitList[0], "score:  ", lineSplitList[8])

                messageChangeTimeTransactionStrList = lineSplitList[0].split(";")
                lenFeature = (len(messageChangeTimeTransactionStrList) - self.transCount) // 2
                priceStrList = messageChangeTimeTransactionStrList[:lenFeature]
                timeStrList = messageChangeTimeTransactionStrList[lenFeature:2*lenFeature]
                transactionStrList = messageChangeTimeTransactionStrList[-self.transCount:]

                resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
                resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
                resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]
                score = float(lineSplitList[8])
                line = fp.readline()
                if score < 0.5:
                    continue
                repeatCount = min(1, abs((1.0 - score)*100.0)//2 )
                while repeatCount > 0:
                    repeatCount -= 1
                    for binCount in range( self.maxFeatureCount - self.minFeatureCount - 1):
                        curCount = binCount + self.minFeatureCount
                        totalCurves = resultsChangeFloat[-curCount:] + resultsTimeFloat[-curCount:]
                        self.featureArr[binCount].append(totalCurves)
                    newTransaction  = resultsTransactionFloat + [resultsChangeFloat[-1], resultsTimeFloat[-1]]
                    assert ( len(newTransaction) == 11 )
                    print(newTransaction)
                    self.transactionFeaturesList.append( newTransaction )
                    self.scoreList.append( 1 if score > 1.0 else 0 )
