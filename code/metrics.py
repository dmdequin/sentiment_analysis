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
	dom = args[1]
	things = args[2:]
	#print(things)
	
	# initialising empty dataframe to concat results
	columns = ['domain', 'trial_type', 'add_data', 'correctly_predicted', 'incorrectly_predicted', 'total_predicted_positives', 'ground_truth_positives', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'recall', 'f1']
	df = pd.DataFrame(columns=columns)
	print_state = True

	for l in things: 		
		if print_state == True:
			print(f'{dom}_{l} data:')
			subject = pd.read_csv(f'data/predictions/{dom}_{l}_preds.csv', header=None, sep=',')
			
			subject = subject.T
			subject = subject[:-1]
			print(f'\nThere are {int(sum(subject[0]))} predicted positives')
			#print(subject.shape)

			subject_true = pd.read_csv(f'data/interim/{dom}_test.csv', header=None)
			#subject_true = subject_true[:1000]
			subject_true = subject_true[1]
			#print(subject_true)
			print(f'There are {sum(subject_true)} true positives')
			#print(sew_full.shape)

			pc, pw, nc, nw, acc = comparerer(subject_true, subject[0])

			
			print(f'There are {pc + nc} correct')
			print(f'There are {pw + nw} incorrect')

			print(f'\nAccuracy:        {acc}%')
			print(f'True positives:  {pc}')
			print(f'True negatives:  {nc}')
			print(f'False positives: {nw}')
			print(f'False negatives: {pw}')

			f1 = f1_score(subject_true, subject[0], average='weighted')
			print(f'F1:              {round(f1, 3)}')

			p = precision_score(subject_true, subject[0], average='weighted')
			print(f'Precision:       {round(p,3)}')

			r = recall_score(subject_true, subject[0], average='weighted')
			print(f'Recall:          {round(r,3)}')


			print('\n' + '-'*50 + '\n')

		# saved the dict as it might be useful
		# dict_metrics = {
		# 	'model' : f'{dom}_{l}',
		# 	'correctly predicted' : pc + nc,
		# 	'incorrectly predicted' : pw + nw,
		# 	'total predicted positives' : int(sum(subject[0])),
		# 	'num of positives in ground truth' : sum(subject_true),
		# 	'TP' : pc,
		# 	'TN' : nc,
		# 	'FP' : nw,
		# 	'FN' : pw,
		# 	'accuracy' : acc,
		# 	'precision' : p,
		# 	'recall' : r,
		# 	'f1' : f1
		# }

		data = {f'{dom}_{l}': [f'{dom}', f'{l[-2:]}', f'{l[-7:-2]}', pc + nc, pw + nw, int(sum(subject[0])), sum(subject_true), pc, nc, nw, pw, acc, p, r, f1]}
		df2 = pd.DataFrame.from_dict(data, orient='index', columns=columns)
		#print(df2)
		df = pd.concat([df, df2], ignore_index = True)
	#print (df)
	
	if len(df['trial_type'].unique()) > 2:
		filename = f'report/metrics/{dom}_mixed_metrics.csv'
	else:
		filename = f'report/metrics/{dom}_{l[-2:]}_metrics.csv'
	# filename = 'test.csv'
	df.to_csv(filename, index=False)