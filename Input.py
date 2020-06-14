import numpy as np

class RiseMinute:
    def __init__(self, riseAndTimeStr):
        riseAndTimeStrPair = riseAndTimeStr.split("|")
        self.rise = riseAndTimeStrPair[0]
        self.time = riseAndTimeStrPair[1]

    def __repr__(self):
        return "Rise:%s,Time:%s" % (self.rise, self.time)


class ReShapedInput:
    def __init__(self, featureCount ):
        self.inputRise = []
        self.inputTime = []
        self.featureCount = featureCount
        self.featureArr = []

    def concanate(self, riseAndTimeList ):
        riseList = list(map( lambda x: float(x.rise), riseAndTimeList ))
        newList = list(map(lambda x: riseList[x:x+self.featureCount],
                                  range( len( riseList ) - self.featureCount)))

        timeList = list(map( lambda x: float(x.time), riseAndTimeList ))
        newTimeList = list(map(lambda x: timeList[x:x+self.featureCount],
                                  range( len( timeList ) - self.featureCount)))

        self.inputRise.extend(newList)
        self.inputTime.extend(newTimeList)

    def toNumpy(self):
        inputSize = len(self.inputRise)
        if np.size(self.featureArr, 0) == inputSize:
            return self.featureArr
        temp = []
        for curIndex in range(inputSize):
            newRow = self.inputRise[curIndex] + self.inputTime[curIndex]
            temp.append(newRow)
        self.featureArr = np.array(temp);
        self.featureArr.reshape(-1, self.featureCount*2)
        return self.featureArr