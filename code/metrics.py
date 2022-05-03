import pandas as pd
from sklearn.metrics import f1_score
import sys

def comparerer(truth, predictions):
	if len(predictions) != len(truth):
		print('Lengths don\'t match!')
		print(len(predictions))
		print(len(truth))
		return 0,0,0

	else:
		correct = 0
		wrong = 0
		for i in range(len(truth)):
			if int(predictions[i]) == int(truth[i]):
				correct += 1
			else:
				wrong += 1

		return correct, wrong, (100*correct)/len(truth)

if __name__ == '__main__':
	args = sys.argv
	print(args)
	things = args[1:]

	for l in things:

		print(f'{l} data:')
		subject = pd.read_csv(f'../data/predictions/{l}_preds.csv', header=None)
		subject = subject.T
		subject = subject[:-1]
		print(f'\nThere are {int(sum(subject[0]))} predicted positive')
		#print(subject.shape)

		subject_true = pd.read_csv(f'../data/interim/{l[:-5]}_test.csv')
		subject_true = subject_true[:1000]
		subject_true = subject_true['label']
		print(f'There are {sum(subject_true)} actual positive')
		#print(sew_full.shape)

		c, w, acc = comparerer(subject_true, subject[0])

		
		print(f'There are {c} correct')
		print(f'There are {w} incorrect')
		print(f'Accuracy is {acc}%')

		f1 = f1_score(subject_true, subject[0], average='weighted')
		print(f'F1 is {round(f1, 3)}')

		print('\n' + '-'*50 + '\n')