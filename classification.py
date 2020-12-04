import sys
import os
import re
import string
import csv
import spacy
import numpy as np


def readInput(input, delimiter):
    dataset = []
    lines = csv.reader(input, delimiter=delimiter)
    for line in lines:
        dataset.append(line)
    return dataset