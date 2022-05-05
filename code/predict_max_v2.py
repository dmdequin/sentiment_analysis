import numpy as np
import json
import pandas as pd

def get_predictions(PROBS, PRED_FILE):
    # Get predictions from the probabilities
    probs = []
    for p in PROBS:
        a = str(p).split(' ')
        t = []
        for u in a:
            if len(u) > 1:
                t.append(float(u[1:-1]))
        probs.append(t)

    probs = np.array(PROBS)
    #print(probs)
    preds = np.argmax(probs, axis=1)
    #print(preds)

    print(f'\npredictions made, writing to {PRED_FILE}')

    with open ('../data/predictions/'+PRED_FILE+'.csv', 'w') as f:
        for i in preds:
            f.writelines(str(i)+',')



if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args[1])
    PROB = args[1]       # probabilities list!
    PRED = args[2]       # prediction file name # 'music_reviews_test_argmax'

    get_predictions(PROB, PRED)