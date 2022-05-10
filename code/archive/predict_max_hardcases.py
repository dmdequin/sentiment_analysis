import numpy as np
import json
import pandas as pd

def get_predictions(PATH):
    # Get predictions from the probabilities
    probs = []
    with open('../data/probabilities/' + PATH, 'r') as f:
        temp = (f.read().split(','))
        for i in temp[:-1]:
            a = i.split(' ')
            t = []
            for u in a:
                if len(u) > 1:
                    t.append(float(u[1:-1]))
            probs.append(t)
    probs = np.array(probs)
    preds = np.argmax(probs, axis=1)

    # convert 0s and 1s into strings
    y_hat = []
    for i in preds:
        if i:
            y_hat.append('positive')
        else: y_hat.append('negative')
    
    # for export 
    test_pred = pd.read_json( '../data/raw/phase2_testData-masked.json.gz', lines=True)
    test_pred['sentiment'] = y_hat

    test_pred = test_pred[['reviewText', 'sentiment', 'category', 'group']].copy()
    
    #Write in a way that codalabs will accept. Thanks to Nicola.
    new = test_pred.to_dict('records')
    test_json=[json.dumps(i)+'\n' for i in new]
    with open ('../data/predictions/phase2_testData'+'.json', 'w') as file:
        file.writelines(test_json)

if __name__ == '__main__':
    import sys
    args = sys.argv
    get_predictions(args[1])