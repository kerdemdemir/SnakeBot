import json
import Input as input
import InputManager as inputManager
import zmq
import numpy as np
import sys

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


transactionBinCount = 7
msecs = 1000

def ReadFileAndCreateReshaper( fileName ):
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    reshaper = inputManager.ReShapeManager()

    for jsonElem in jsonDictionary:
        reshaper.addANewCurrency(jsonElem,msecs,transactionBinCount,False)
    file.close()
    return  reshaper

def AddExtraToShaper ( fileName, shaper, IsTransactionOnly):
    file = open(fileName, "r")
    jsonDictionary = json.load(file)

    for jsonElem in jsonDictionary:
        shaper.addANewCurrency(jsonElem,msecs,transactionBinCount,IsTransactionOnly)
    file.close()

trainingReshaper = ReadFileAndCreateReshaper("learning_23_06.txt")
AddExtraToShaper("learning24_06.txt",trainingReshaper, True)
AddExtraToShaper("learning_25_06.txt",trainingReshaper, True)
AddExtraToShaper("learning_26_29.txt",trainingReshaper, True)
AddExtraToShaper("learning_29_30.txt",trainingReshaper, True)
AddExtraToShaper("learning_30.txt",trainingReshaper, True)
AddExtraToShaper("learning_30_1.txt",trainingReshaper, True)
AddExtraToShaper("learning_1_3.txt",trainingReshaper, True)
AddExtraToShaper("learning_3_7.txt",trainingReshaper, True)
AddExtraToShaper("learning_7_9.txt",trainingReshaper, True)
AddExtraToShaper("learning_9_10.txt",trainingReshaper, False)

print("All added now scores")
#trainingReshaper.transactionHelper.Print()
trainingReshaper.assignScores()
print("Assigned scores")
sys.stdout.flush()

numpyArr = trainingReshaper.toTransactionFeaturesNumpy(transactionBinCount)
mlpTransaction = MLPClassifier(hidden_layer_sizes=(transactionBinCount, transactionBinCount, transactionBinCount), activation='relu',
                                              solver='adam', max_iter=500)
X = numpyArr
y = trainingReshaper.toTransactionResultsNumpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
mlpTransaction.fit(X_train, y_train)

predict_test = mlpTransaction.predict(X_test)
print(" Transactions : ")
print(confusion_matrix(y_test, predict_test))

mlpList = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - inputManager.ReShapeManager.minFeatureCount)]
mlpScalerList = [[] for _ in range(inputManager.ReShapeManager.maxFeatureCount - inputManager.ReShapeManager.minFeatureCount)]

for binCount in range (inputManager.ReShapeManager.minFeatureCount, inputManager.ReShapeManager.maxFeatureCount):
    #numpyArr = trainingReshaper.toTransactionFeaturesNumpy(binCount)
    #X_train = trainingReshaper.toTransactionFeaturesNumpy(binCount,transactionBinCount)
    #y_train = trainingReshaper.toResultsNumpy(binCount)
    #X_test = trainingReshaper2.toTransactionFeaturesNumpy(binCount,transactionBinCount)
    #y_test = trainingReshaper2.toResultsNumpy(binCount)

    #print( X.shape, " ", y.shape,  X.shape, " ", y_1.shape,  X_2.shape, " ", y_2.shape)
    curIndex = binCount - inputManager.ReShapeManager.minFeatureCount
    numpyArr = trainingReshaper.toFeaturesNumpy(binCount)
    mlpScalerList[curIndex] = preprocessing.StandardScaler().fit(numpyArr)
    X = mlpScalerList[curIndex].transform(numpyArr)
    y = trainingReshaper.toResultsNumpy(binCount)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
    y_test = trainingReshaper.toTestResultNumpy(X_test,binCount)
    nodeSize = min(8, binCount*2)
    mlpList[curIndex] = MLPClassifier(hidden_layer_sizes=(nodeSize,nodeSize,nodeSize), activation='relu', solver='adam', max_iter=500)
    mlpList[curIndex].fit(X_train,y_train)
    predict_test = mlpList[curIndex].predict(X_test)
    print( " Curves : ")
    print( confusion_matrix(y_test,predict_test))
    sys.stdout.flush()


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("ipc:///tmp/peakLearner")

while True:
    #  Wait for next request from client
    message = socket.recv_string(0, encoding='ascii')
    print("Received request: %s" % message)
    messageChangeTimeTransactionStrList = message.split(";")
    priceStrList = messageChangeTimeTransactionStrList[0].split(",")
    timeStrList = messageChangeTimeTransactionStrList[1].split(",")
    transactionStrList = messageChangeTimeTransactionStrList[2].split(",")

    resultsChangeFloat = [float(messageStr) for messageStr in priceStrList]
    resultsTimeFloat = [float(timeStr) for timeStr in timeStrList]
    resultsTransactionFloat = [float(transactionStr) for transactionStr in transactionStrList]

    resultStr = ""

    totalFeatures = resultsTransactionFloat[:transactionBinCount + 1]
    print("I will predict: ", totalFeatures)
    npTotalFeatures = np.array(totalFeatures)
    npTotalFeatures = npTotalFeatures.reshape(1, -1)
    predict_test = mlpTransaction.predict_proba(npTotalFeatures)
    curResultStr = str(predict_test) + ";"
    resultStr += curResultStr
    for binCount in range (inputManager.ReShapeManager.maxFeatureCount-inputManager.ReShapeManager.minFeatureCount-1):
        curCount = binCount + inputManager.ReShapeManager.minFeatureCount
        totalCurves = resultsChangeFloat[-curCount:] + resultsTimeFloat[-curCount:]
        npTotalCurves = np.array(totalCurves)
        npTotalCurves = npTotalCurves.reshape(1,-1)
        npTotalCurvesScaled = mlpScalerList[binCount].transform(npTotalCurves)
        print("I will predict the curves: ", totalCurves)
        predict_test = mlpList[binCount].predict_proba(npTotalCurvesScaled)
        curResultStr = str(predict_test) + ";"
        resultStr += curResultStr

    resultStr = resultStr[:-1]
    print("Results are: " , resultStr)

    #  Send reply back to client
    socket.send_string(resultStr, encoding='ascii')
