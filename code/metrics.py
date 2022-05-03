import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sys

def comparerer(truth, predictions):
	if len(predictions) != len(truth):
		print('Lengths don\'t match!')
		print(len(predictions))
		print(len(truth))
		return 0,0,0

	else:
		pos_correct = 0
		pos_wrong = 0
		neg_correct = 0
		neg_wrong = 0

		for i in range(len(truth)):
			if int(predictions[i]) == int(truth[i]):
				if int(truth[i]) == 1:
					pos_correct += 1
				else:
					neg_correct += 1
			else:
				if int(truth[i]) == 1:
					pos_wrong += 1
				else:
					neg_wrong +=1

		return pos_correct, pos_wrong, neg_correct, neg_wrong, (100*(pos_correct+neg_correct))/len(truth)

if __name__ == '__main__':
	args = sys.argv
	#print(args)
	things = args[1:]

	for l in things:

		print(f'{l} data:')
		subject = pd.read_csv(f'../data/predictions/{l}_preds.csv', header=None)
		subject = subject.T
		subject = subject[:-1]
		print(f'\nThere are {int(sum(subject[0]))} predicted positives')
		#print(subject.shape)

		subject_true = pd.read_csv(f'../data/interim/{l[:-5]}_test.csv')
		#subject_true = subject_true[:1000]
		subject_true = subject_true['label']
		print(f'There are {sum(subject_true)} true positives')
		#print(sew_full.shape)

		pc, pw, nc, nw, acc = comparerer(subject_true, subject[0])

		
		print(f'There are {pc + nc} correct')
		print(f'There are {pw + nw} incorrect')

		print(f'\nAccuracy:        {acc}%')
		print(f'True positives:  {pc}')
		print(f'True negatives:  {nc}')
		print(f'False positives: {pw}')
		print(f'False negatives: {nw}')

		f1 = f1_score(subject_true, subject[0], average='weighted')
		print(f'F1:              {round(f1, 3)}')

		p = precision_score(subject_true, subject[0], average='weighted')
		print(f'Precision:       {round(p,3)}')

		r = recall_score(subject_true, subject[0], average='weighted')
		print(f'Recall:          {round(r,3)}')


		print('\n' + '-'*50 + '\n')