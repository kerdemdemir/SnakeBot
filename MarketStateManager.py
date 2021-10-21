import bisect
import time
from collections import deque

class PeakStateAndTime:
    def __init__(self, time, isBottom):
        self.time =  time
        self.isBottom = isBottom

    def __lt__(self, other):
        return self.time < other.time

    def __repr__(self):
        return "Time:%i,Bottom:%i" % (self.time, self.isBottom)

class MarketStateManager:
    def __init__(self):
        self.stateList = []
        self.curStateList = deque([])
        self.curUpDowns = [ 0, 0, 0, 0]
        self.durationList = [900, 3600]

    def add(self, isRise, timeInSec):
        self.stateList.append(PeakStateAndTime( timeInSec, isRise))

    def addRecent(self, isRise):
        curSeconds = int(time.time())
        newStateAndTime = PeakStateAndTime(curSeconds, isRise)
        self.curStateList.append(newStateAndTime)
        popCount = 0
        for elem in self.curStateList:
            if curSeconds - elem.time > 21600:
                popCount += 1
            else :
                break
                
        for i in range (popCount):
            self.curStateList.popleft()
            
        for index in range( len(self.durationList) ):
            curDuration = self.durationList[index]
            self.curUpDowns[index*2] = self.getCount(True, curDuration,curSeconds)
            self.curUpDowns[index*2+1] = self.getCount(False, curDuration, curSeconds)
        print("New ups and downs " , self.curUpDowns)


    def getCount(self, isRise, duration, curTime ):
        filteredList = filter(lambda x: curTime - x.time < duration and x.isBottom == isRise, self.curStateList)
        return len(list(filteredList))

    def sort(self):
        self.stateList = sorted(self.stateList, key=lambda l: l.time)

    def getNowAndBuyState(self, timeDiffInSec ):
        epoch_time = int(time.time())
        timeInSec = epoch_time - timeDiffInSec
        buyTimeResult = self.getState(timeInSec)
        return self.curUpDowns[0:2] + buyTimeResult[0:2]

    def getState(self, timeInSecs ):
        returnList = []
        curTime = PeakStateAndTime(timeInSecs, False)

        stopIndex = bisect.bisect_left(self.stateList, curTime)
        #print(stopIndex, " ", curTime)
        for duration in self.durationList:
            newTime = PeakStateAndTime(timeInSecs - duration, False)
            startIndex = bisect.bisect_left(self.stateList, newTime )

            topCount = 0
            downCount = 0
            for elem in self.stateList[startIndex:stopIndex+1]:
                if elem.isBottom:
                    downCount += 1
                else:
                    topCount += 1

            returnList.append(downCount)
            returnList.append(topCount)
        return returnList


