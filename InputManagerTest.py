import unittest
import InputManager as inputManager
import Input as input

class MyTestCase(unittest.TestCase):
    def testInit(self):
        self.reshapeManager = inputManager.ReShapeManager()
        self.assertEqual(len(self.reshapeManager.inputs), self.reshapeManager.maxFeatureCount - self.reshapeManager.minFeatureCount)
        self.assertEqual(len(self.reshapeManager.scoreList),
                         self.reshapeManager.maxFeatureCount - self.reshapeManager.minFeatureCount)
        self.__testAddingALine()
        self.__testNumpyConversion()
        #self.__testAddingALine();

    def __testAddingALine(self):
        testLine = "14.28876093|2043,-4.39326082|93,6.20525060|801,-5.07702322|90,4.78147988|750,-7.91864994|1350,19.15910177|4110,-6.47396082|3720,3.36270872|2940,-3.84972171|1260,9.70112690|2160,-3.62567369|4650,3.07017544|480,-6.62768031|4110,3.35380255|420,-4.70004724|180,10.82500000|540"
        riseAndTimeStrList = testLine.split(",")
        riseAndTimeList = list(map(lambda x: input.RiseMinute(x), riseAndTimeStrList))
        self.reshapeManager.addLinePeaks(riseAndTimeList)
        self.assertEqual( len(self.reshapeManager.inputs[0].inputRise), len(riseAndTimeStrList) - 3)
        self.__testResultingScores()

    def __checkingScores(self):
        self.reshapeManager.resetScores()
        self.assertEqual(len(self.reshapeManager.inputs[0].inputRise), len(self.reshapeManager.scoreList[0]))
        self.assertEqual(len(self.reshapeManager.scoreList),
                         self.reshapeManager.maxFeatureCount - self.reshapeManager.minFeatureCount)
        self.__testResultingScores()

    def __testResultingScores(self):
        self.reshapeManager.assignScores()

    def __testNumpyConversion(self):
        numpyArr = self.reshapeManager.toFeaturesNumpy(3)
        self.assertEqual(numpyArr.shape[1], 6)
        self.assertEqual(numpyArr.shape[0], 14)
        scores = self.reshapeManager.toResultsNumpy(3)
        print(scores)
        self.assertEqual(numpyArr.shape[0], scores.shape[0])

if __name__ == '__main__':
    unittest.main()
