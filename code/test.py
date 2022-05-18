from model_run_v2 import *
from predict_max_v2 import *
import sys

if __name__ == '__main__':
    args = sys.argv
    dom = args[1]
    vari = args[2]
    thing = ['00010', '00100', '01000', '10000']

    for t in thing:
        TRIAL = f'{dom}_' + t + f'{vari}' ##We need a consistent naming convention here.

        MODEL = 'model' + TRIAL + '.pkl'
        TEST_FILE = TRIAL[:-8] + '_test.csv'

        print(MODEL)
        print(TEST_FILE)

        PROBS_NAME = TRIAL + '_probs'
        PREDS_NAME = TRIAL + '_preds'


        probs = make_probs(MODEL, TEST_FILE, PROBS_NAME)
        print(len(probs))
        print('\ngetting predictions')
        get_predictions(probs, PREDS_NAME)