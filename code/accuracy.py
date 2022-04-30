import pandas as pd

def comparerer(predictions, truth):
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


sew = pd.read_csv('../data/predictions/sew_base_preds.csv', header=None)
sew = sew.T
sew = sew[:-1]
print(f'There are {sum(sew[0])} predicted positive')
#print(sew.shape)

sew_full = pd.read_csv('../data/interim/sew_test.csv')
sew_full = sew_full[:1000]
sew_full = sew_full['label']
print(f'There are {sum(sew_full)} actual positive')
#print(sew_full.shape)

c, w, acc = comparerer(sew[0], sew_full)

print(f'Sewing scores:')
print(f'There are {c} correct')
print(f'There are {w} incorrect')
print(f'Accuracy is {acc}%')

print('\n' + '-'*50 + '\n')
games = pd.read_csv('../data/predictions/games_base_preds.csv', header=None)
games = games.T
games = games[:-1]
print(f'There are {sum(games[0])} predicted positive')
#print(games.shape)

games_full = pd.read_csv('../data/interim/games_test.csv')
games_full = games_full[:1000]
games_full = games_full['label']
print(f'There are {sum(games_full)} actual positive')
#print(games_full.shape)

c, w, acc = comparerer(games[0], games_full)

print(f'Games scores:')
print(f'There are {c} correct')
print(f'There are {w} incorrect')
print(f'Accuracy is {acc}%')