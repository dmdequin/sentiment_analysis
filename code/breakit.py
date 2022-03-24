import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
import spacy
from random import randint
import json

nlp = spacy.load('en_core_web_sm')

TRAIN = '../data/interim/train.csv'

input_data = []
label = []
c = 0
n = 500
s = 20
with open(TRAIN, mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        if c < n-s:
            c+=1
        elif c < n:
            input_data.append(lines[0])
            label.append(int(lines[1]))
            c+=1
        else:
            break

print(len(input_data))
    
def splitsies(para):
    punct = re.compile('[.?!:]')
    t = punct.split(para)
    spl= []
    for i in t:
        if len(i) > 0:
            spl.append(i)
        
    return spl

def randomess(sent):
	r = randint(1,3)
	if r == 1:
		if len(sent) > 2:
			#print('1')
			changes = 'typos'
			return Perturb.add_typos(sent), changes
		else: return sent, 'punct'

	if r == 2:
		if len(sent) > 2:
			#print('2')
			changes = 'typos'
			temp = Perturb.add_typos(sent)
			return Perturb.add_typos(temp), changes
		else: return sent, 'punct'

	if r == 3:
		#print('3')
		if randint(0,1):
			changes = 'punct'
			return sent + '!!!', changes
		elif randint(0,1):
			changes = 'punct'
			return sent + '.', changes
		else:
			return sent,'punct'

 
def messy(review):
	temp = splitsies(review)
	n = len(temp)
	final = ''
	ch = set()
	for j in range(n):
		sent = temp[j]
		#print(sent)
		if randint(0,1):
			#print('choice 1')
			a, b = randomess(sent)
			final += a
			ch.update([b])

		else:
			#print('choice 2')
			a,b = randomess(sent)
			c,d= randomess(a)
			ch.update([b])
			ch.update([d])
	return final, ch


thing = []
chan = []
for i in input_data[:]:
	#print(i)
	t, c = messy(i)
	thing.append(t)
	chan.append(str(c))
	
	#print(chan[-1])
	#print(thing[-1])
	#print('-'*50)

labels = []
for i in label:
    if i:
        labels.append('positive')
    else: labels.append('negative')


output = []
for i in range(len(input_data)):
    dicti = {}
    dicti['reviewText'] = thing[i]
    dicti['sentiment'] = labels[i]
    dicti['category'] = chan[i]
    output.append(dicti)

test_json=[json.dumps(i)+'\n' for i in output]
with open ('../data/predictions/sanna_dump.json', 'w') as file:
    file.writelines(test_json)