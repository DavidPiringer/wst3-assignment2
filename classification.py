import sys
import os
import re
import string
import csv
import spacy
import numpy as np
from sklearn.model_selection import train_test_split

def readInput(file, delimiter):
    input = open(str(file), 'r')
    dataset = []
    lines = csv.reader(input, delimiter=delimiter)
    for line in lines:
        dataset.append(line)
    input.close()
    return dataset

# constant variables
training_ratio = 0.3

print('## Processing input ...\n')

ctd = readInput('CtD.csv', ';')
ttd = readInput('ttd.csv', ',')

print('length of ctd = ' + str(len(ctd)))
print('length of ttd = ' + str(len(ttd)))

print('\n## Splitting training and test dataset... \n')

# split into training an test data
# 70% for training
# 30% for testing
ctd_train, ctd_test, ttd_train, ttd_test = train_test_split(ctd, ttd, test_size=training_ratio, random_state=128)

print('length of training ctd = ' + str(len(ctd_train)))
print('length of testing ctd = ' + str(len(ctd_test)))
print('length of training ttd = ' + str(len(ttd_train)))
print('length of testing ttd = ' + str(len(ttd_test)))

print('\nfirst member of training dataset: ')
print('    ' + str(ctd_train[0]))
print('\nfirst member of testing dataset: ')
print('    ' + str(ctd_test[0]))

print('\n## Evaluating classification performance using the Rocchio classifier ... \n')