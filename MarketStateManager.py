import bisect
import TransactionHelper
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
        self.curUpDowns = [0, 0, 0, 0 , 0, 0]
        self.durationList = [60, 300, 21600]

    def add(self, transactionPeakHelper):
        self.stateList.append(PeakStateAndTime( transactionPeakHelper.peakTimeSeconds, transactionPeakHelper.isBottom))

    def addRecent(self, isBottom):
        curSeconds = int(time.time())
        newStateAndTime = PeakStateAndTime(curSeconds, isBottom)
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


    def getCount(self, isBottom, duration, curTime ):
        filteredList = filter(lambda x: curTime - x.time < duration and x.isBottom == isBottom, self.curStateList)
        return len(list(filteredList))


    def sort(self):
        self.stateList = sorted(self.stateList, key=lambda l: l.time)

    def getState(self, timeInSecs ):
        returnList = []
        curTime = PeakStateAndTime(timeInSecs, False)

        stopIndex = bisect.bisect_left(self.stateList, curTime)
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

