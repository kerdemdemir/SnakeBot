import unittest
import InputManager as inputManager
import Input as input
import json
import TransactionHelper

class MyTestCase(unittest.TestCase):
    def testInit(self):
        self.reshapeManager = inputManager.ReShapeManager()
        file = open("transaction.json", "r")
        self.jsonData = json.load(file)
        self.__testTransactionData()
        self.__testTransactionPeak()
        self.__testTransactionPeak2()
        file.close()

    def __testTransactionData(self):
        twoDataLine = self.jsonData[0]["transactionList"][19]
        singleData = TransactionHelper.TransactionData()
        for elem in twoDataLine:
            singleData.AddData(elem)
        # TransactionData, self.totalBuy = 0.0, self.totalSell = 0.0,self.transactionCount = 0.0,self.score = 0
        self.assertNotEqual( singleData.totalBuy, 0)
        self.assertNotEqual( singleData.totalSell, 0)
        self.assertEqual( singleData.transactionCount, 1)

    def __testTransactionPeak(self):
        manyDataLine = self.jsonData[1]["transactionList"][36]
        peakData = self.jsonData[1]["peak"].split(",")[36]
        riseMinute = input.RiseMinute(peakData)
        singleData = TransactionHelper.TransactionPeakHelper(manyDataLine, 3000, riseMinute.rise < 0.0)
        self.assertEqual( len(singleData.dataList), 8 )
        singleData.AssignScores()

    def __testTransactionPeak2(self):
        manyDataLine = self.jsonData[1]["transactionList"][40]
        print( len(manyDataLine))
        peakData = self.jsonData[1]["peak"].split(",")[40]
        riseMinute = input.RiseMinute(peakData)
        singleData = TransactionHelper.TransactionPeakHelper(manyDataLine, 1000, riseMinute.rise < 0.0)
        singleData.AssignScores()
        print( *singleData.patternList[0].keys() )

if __name__ == '__main__':
    unittest.main()
