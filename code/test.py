from model_run_v2 import *
from predict_max_v2 import *

TRIAL = 'games_base' ##We need a consistent naming convention here.

'''MODEL = 'model_' + TRIAL + '.pkl'
TEST_FILE = TRIAL[:-4] + '_test.csv' '''

MODEL = 'model_base.pkl'
TEST_FILE = 'sew_test.csv' 
PROBS_NAME = TRIAL + '_probs'
PREDS_NAME = TRIAL + '_preds'


probs = make_probs(MODEL, TEST_FILE, PROBS_NAME)
print('\ngetting predictions')
get_predictions(probs, PREDS_NAME)