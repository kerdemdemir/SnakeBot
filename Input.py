import numpy as np
import itertools
import bisect

class RiseMinute:
    def __init__(self, riseAndTimeStr):
        riseAndTimeStrPair = riseAndTimeStr.split("|")
        self.rise = float(riseAndTimeStrPair[0])
        self.time = riseAndTimeStrPair[1]

    def __repr__(self):
        return "Rise:%f,Time:%s" % (self.rise, self.time)

class InputRiseSorter:

    def __init__(self, featureCount ):
        self.sortedPriceList = []
        self.sortedPriceKeys = []
        self.featureCount = featureCount

    def add(self,inputRise):
        self.sortedPriceList = inputRise
        self.sortedPriceList = sorted(self.sortedPriceList, key=lambda l: l[0])
        self.sortedPriceKeys = [elem[0] for elem in self.sortedPriceList]

    def getIndex(self, elem):
        return bisect.bisect(self.sortedPriceKeys, elem)


class ReShapedInput:

    def __init__(self, featureCount ):
        self.inputRise = []
        self.inputSorter = InputRiseSorter(featureCount)
        self.inputTime = []
        self.featureCount = featureCount

    def concanate(self, riseAndTimeList):
        riseList = list(map( lambda x: float(x.rise), riseAndTimeList ))
        newList = list(map(lambda x: riseList[x:x+self.featureCount],
                                  range( len( riseList ) + 1 - self.featureCount)))


        timeList = list(map( lambda x: float(x.time), riseAndTimeList ))
        newTimeList = list(map(lambda x: timeList[x:x+self.featureCount],
                                  range( len( timeList ) + 1 - self.featureCount)))

        self.inputRise.extend(newList)
        self.inputTime.extend(newTimeList)

    def getSorter(self):
        if ( len(self.inputSorter.sortedPriceKeys) == 0 ):
            self.inputSorter.add(self.inputRise)
            del self.inputRise
        return self.inputSorter
