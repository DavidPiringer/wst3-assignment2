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


ds = readInput('CtD.csv', ';')
print(ds)