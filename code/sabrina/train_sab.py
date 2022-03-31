# basic reqs
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F

# pre processing reqs 
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# model reqs
import torch.nn as nn
from transformers import BertModel
import time



def pre_process(path, N):
    '''
    input: path to json file
    output: bert encodings
    JSON -> df -> bert tokens -> bert embeddings
    '''
    df = pd.read_json(path, lines=True)

    df['concatSummaryReview'] = df['summary'] + ' ' + df['reviewText']
    df['concatSummaryReview'] = df['concatSummaryReview'].str.lower().fillna('[NA]')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # get the labels
    sentiment_binary = {
    'positive' : 1,
    'negative' : 0
    }

    df['sentiment_binary'] = df['sentiment'].map(sentiment_binary)
    labels = torch.tensor(df['sentiment_binary'])

    # lists to store outputs of bert tokeniser 
    input_ids = []
    attention_masks = []

    MAX_LEN = 512
    BATCH_SIZE = 2 # For fine-tuning BERT, the authors recommend a batch size of 16 or 32

    for i in range(N):
        text = df['concatSummaryReview'][i]
        tokens = tokenizer.tokenize(text)
        encoded_plus = tokenizer.encode_plus(
            text=tokens,                  # Preprocess sentence
            add_special_tokens=True,    # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,         # Max length to truncate/pad
            padding='max_length',       # Pad sentence to max length
            return_attention_mask=True, # Return attention mask
            truncation = True
            # return_tensors='pt',        # Return PyTorch tensor
        )

        input_ids.append(encoded_plus.get('input_ids'))
        attention_masks.append(encoded_plus.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    # from here will be different for the test data
    # and for the sata used for predictions - might need to split into 2 functions

    # Create the DataLoader to be used as input in the model
    labels = labels[0:N] # to limit the labels for testing purposes
    data = TensorDataset(input_ids, attention_masks, labels)
    sampler = RandomSampler(data)
    loader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)

    return loader, labels

# Create the BertClassfier class
class BertClassifier(nn.Module):
    '''Bert Model for Classification Tasks.
    '''
    def __init__(self, freeze_bert=False):
        '''
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        '''
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        '''
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        '''
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits

# defining optimiser
def initialize_model(epochs=4):
    '''
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    '''
    from transformers import AdamW, get_linear_schedule_with_warmup
    
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_labels) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    import random
    '''
    Set seed for reproducibility.
    '''
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Specify loss function
LOSS_FN = nn.CrossEntropyLoss()

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    '''
    Train the BertClassifier model.
    '''

    # Start training loop
    print("Start training...\n")

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1

            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = LOSS_FN(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    '''
    inputs: model and dataset to be evaluated
    outputs:list of losses (?) and accuracy

    After the completion of each training epoch, measure the model's performance
    on our validation set.
    '''

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = LOSS_FN(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


N = 50

TRAIN = '../../data/raw/music_reviews_train.json'
DEV = '../../data/raw/music_reviews_dev.json'
TEST = '../../data/raw/music_reviews_test_masked.json'

train_loader, train_labels = pre_process(TRAIN, N)
dev_loader, dev_labels = pre_process(DEV, N)
print('Pre process done.')

EPOCHS = 2

set_seed(42)    # Set seed for reproducibility
bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)

print('Starting training.')
train(bert_classifier, train_loader, dev_loader, epochs=EPOCHS, evaluation=True)
print('Training finished.')

# save model
import pickle

model = bert_classifier
pickle.dump(model, open('sab_model.pkl', 'wb'))
print('Model saved.')