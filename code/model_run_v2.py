import pickle
import sys
from baseline import *
import torch

def make_probs(MODEL,TEST_FILE, PROBS_FILE):
    print('making probs')
    batch_size = 2

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        #print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        #print('Device name:', torch.cuda.get_device_name(0))
    else:
        #print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    X_test = loader('/data/dissimilar/'+TEST_FILE) 
    X_test = X_test[:-1]
    ### Remove this line to run predictions on the entire test file. It will take AGES!!
    '''X_test = X_test[:1000]'''
    ###

    print(len(X_test))

    # Fix X_test so that it is a single list of strings
    test = []
    for i in range(len(X_test)):
        test.append(str(X_test[i][0]))
    X_test = test

    # tokenise with BERT
    from transformers import BertTokenizer

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # BERT model defs
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    set_seed(42)    # Set seed for reproducibility

    bert_classifier = pickle.load(open('models/' + MODEL, 'rb'))
    
    print('model loaded')

    # inference
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(X_test)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    import torch.nn.functional as F

    def bert_predict(model, test_dataloader):
        """Perform a forward pass on the trained BERT model to predict probabilities
        on the test set.
        """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        model.eval()

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)
            all_logits.append(logits)
        
        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        return probs

    # Compute predicted probabilities on the test set
    probs = bert_predict(bert_classifier, test_dataloader)

    print(f'\nprobs calculated, writing to {PROBS_FILE}')
    with open ('../data/probabilities/' + PROBS_FILE + '.csv', 'w') as f:
        for i in probs:
            f.writelines(str(i)+',')
    
    return probs 


if __name__ == '__main__':
    args = sys.argv
    TEST_FILE = args[1]  # 'test.csv'           # test data to be run through the model
    MODEL = args[2]      # 'model_ALL_ALL.pkl'  # model that is being used
    PROBS_FILE = args[3] # 'probs_music'    # name to save the probabilities

    probs = make_probs(MODEL,TEST_FILE, PROBS_FILE)