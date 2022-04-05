'''
Usage
python3 compare.py ../data/raw/group12.json.gz ../data/predictions/hardcases_predictions.json'''

def compare(one, two):
	import pandas as pd
	labelled = pd.read_json(one, lines=True)
	pred = pd.read_json(two, lines=True)
	incorrect = labelled[labelled['sentiment']!=pred['sentiment']]
	incorrect.columns = ['reviewText', 'True_sentiment', 'category']

	incorrect.to_csv('../data/interim/incorrect.csv', index=False)
	print(f'There are {incorrect.shape[0]} labels. Output to incorrect.csv')


if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args)
    compare(args[1], args[2])