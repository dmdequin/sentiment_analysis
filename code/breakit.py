import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAIN = '../data/interim/train.csv'

input_data = []
label = []
c = 0
with open(TRAIN, mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        if c < 100:
            input_data.append(lines[0])
            label.append(int(lines[1]))
            c+=1
        else:
            break
print(len(input_data))