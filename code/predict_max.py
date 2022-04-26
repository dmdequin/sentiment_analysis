import numpy as np
import json
import pandas as pd

def get_predictions(PROBS_FILE, TEST_FILE, PRED_FILE):
    # Get predictions from the probabilities
    probs = []
    with open('../data/probabilities/' + PROBS_FILE, 'r') as f:
        temp = (f.read().split(','))
        for i in temp[:-1]:
            #print('i', i)
            a = i.split(' ')
            #print(len(a))
            t = []
            for u in a:
                if len(u) > 1:
                    t.append(float(u[1:-1]))
            #print((t))
            probs.append(t)
    #print(probs)
    probs = np.array(probs)
    #print(probs)
    preds = np.argmax(probs, axis=1)
    #print(preds)

    # convert 0s and 1s into strings
    y_hat = []
    for i in preds:
        if i:
            y_hat.append('positive')
        else: y_hat.append('negative')
        
    # for export 
    test_pred = pd.read_json( '../data/raw/'+TEST_FILE, lines=True)
    test_pred = test_pred[-100:]
    test_pred['sentiment'] = y_hat
    
    #Write in a way that codalabs will accept. Thanks to Nicola.
    new = test_pred.to_dict('records')
    test_json=[json.dumps(i)+'\n' for i in new]
    with open ('../data/predictions/'+PRED_FILE+'.json', 'w') as file:
        file.writelines(test_json)


if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args[1])
    PROB = args[1]       # probabilities file
    TEST_DATA = args[2]  # test data            # 'music_reviews_test_masked.json.gz'
    PRED = args[3]       # prediction file name # 'music_reviews_test_argmax'
    get_predictions(PROB, TEST_DATA, PRED)