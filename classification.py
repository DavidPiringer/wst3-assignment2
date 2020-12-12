import sys
import os
import re
import string
import csv
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier

# constant variables
training_ratio = 0.3
classes = [0, 1, 2, 3]


def readInput(file, delimiter):
    input = open(str(file), 'r')
    dataset = []
    lines = csv.reader(input, delimiter=delimiter)
    for line in lines:
        dataset.append(line)
    input.close()
    return dataset


def getClasses(ctd):
    c = []
    for line in ctd:
        i = classes[0] - 1
        for row in line:
            if (row == '1'):
                c.append(i)
            else:
                i += 1
    return c


def getAllConfusionMatrices(targetClass, predictions, targets):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for idx in range(0, min(len(predictions), len(targets))):
        if (targets[idx] == targetClass):  # A0
            if (targets[idx] == predictions[idx]):  # A0 P0
                tp += 1
            else:  # A0 P1
                fn += 1
        else:
            if (targets[idx] != predictions[idx]):  # A1
                fp += 1  # A1 P0
            else:
                tn += 1  # A1 P1
    return (tp, tn, fp, fn)


def calcPrecision(truePositive, falsePositive):
    return truePositive / (truePositive + falsePositive)


def calcRecall(truePositive, falseNegative):
    return truePositive / (truePositive + falseNegative)


def calcFMeasure(precision, recall):
    if (precision > 0 or recall > 0):
        return 2 * (precision * recall) / (precision + recall)
    return 0.0


def printClassStats(targetClass, predictions, targets):
    tp, tn, fp, fn = getAllConfusionMatrices(targetClass, predictions, targets)
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    fmeasure = calcFMeasure(precision, recall)
    print(f'------- CLASS {targetClass} -------')
    print(f'  Precision: {precision}')
    print(f'     Recall: {recall}')
    print(f'  F-Measure: {fmeasure}')
    print(f'-----------------------\n')


def printPredictionResults(classes_train_predicted, classes_test_predicted):
    print(f'########## TRAIN ##########')
    printClassStats(classes[0], classes_train_predicted, classes_test)
    printClassStats(classes[1], classes_train_predicted, classes_test)
    printClassStats(classes[2], classes_train_predicted, classes_test)
    printClassStats(classes[3], classes_train_predicted, classes_test)

    print(f'########## TEST ##########')
    printClassStats(classes[0], classes_test_predicted, classes_test)
    printClassStats(classes[1], classes_test_predicted, classes_test)
    printClassStats(classes[2], classes_test_predicted, classes_test)
    printClassStats(classes[3], classes_test_predicted, classes_test)


########################################################################
print('## Processing input ...\n')

ctd = readInput('CtD.csv', ';')
ttd = readInput('ttd.csv', ',')

print('length of ctd = ' + str(len(ctd)))
print('length of ttd = ' + str(len(ttd)))

print('\n## Splitting training and test dataset... \n')

# split into training an test data
# 70% for training
# 30% for testing
ttd_train, ttd_test, ctd_train, ctd_test = train_test_split(ttd, ctd, test_size=training_ratio, random_state=128)

print('length of training ctd = ' + str(len(ctd_train)))
print('length of testing ctd = ' + str(len(ctd_test)))
print('length of training ttd = ' + str(len(ttd_train)))
print('length of testing ttd = ' + str(len(ttd_test)))

print('\nfirst member of training dataset: ')
print('    ' + str(ctd_train[0]))
print('\nfirst member of testing dataset: ')
print('    ' + str(ctd_test[0]))


classes_train = getClasses(ctd_train)
classes_test = getClasses(ctd_test)

########################################################################
print('\n## Evaluating classification performance using the Rocchio classifier ... \n')
print(f'~~~~~~~ Rocchio (euclidean) ~~~~~~~')
rocchio_clf = NearestCentroid(metric='euclidean')  # manhattan
rocchio_clf.fit(np.array(ttd_train).astype(np.float64), classes_train)

classes_train_predicted = rocchio_clf.predict(np.array(ttd_train).astype(np.float64))
classes_test_predicted = rocchio_clf.predict(np.array(ttd_test).astype(np.float64))

printPredictionResults(classes_train_predicted, classes_test_predicted)

#      Pos Neg
#      A0  A1
# P0:   TP  FP  Ja
# P1:   FN  TN  Nein
#      Ja  Nein

########################################################################
print('\n## Evaluating classification performance using the kNN classifier ... \n')
print(f'~~~~~~~ kNN (n_neighbors = 3) ~~~~~~~')
# experiment with different values vor n_neighbors
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(np.array(ttd_train).astype(np.float64), classes_train)

classes_train_predicted = knn_clf.predict(np.array(ttd_train).astype(np.float64))
classes_test_predicted = knn_clf.predict(np.array(ttd_test).astype(np.float64))

printPredictionResults(classes_train_predicted, classes_test_predicted)



print('\n## Evaluating classification performance using the kNN classifier ... \n')

print(f'~~~~~~~ Naive Bayes ~~~~~~~')

gnb = GaussianNB()
gnb.fit(np.array(ttd_train).astype(np.float64), classes_train)

classes_train_predicted = gnb.predict(np.array(ttd_train).astype(np.float64))
classes_test_predicted = gnb.predict(np.array(ttd_test).astype(np.float64))

printPredictionResults(classes_train_predicted, classes_test_predicted)

########################################################################