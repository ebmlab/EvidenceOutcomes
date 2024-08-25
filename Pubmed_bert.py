import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os 
from torch import cuda
from scipy import stats
from torch.nn import CrossEntropyLoss
import random 

from seqeval.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import classification_report as clr

train_name = "#train Name"
data = pd.read_csv("#path/to/train_data")

test_data = data 
test_data['length'] = test_data['sentence'].apply(lambda x: len(x.split()))
max_length = test_data['length'].max()
max_length

label2id = {'O': 0, 'B-Outcome': 1, 'I-Outcome': 2, 'X': 3, "[CLS]": 4, "[SEP]": 5}
id2label = {v: k for k, v in label2id.items()}

#Parameter settings
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 1
EPOCHS = 10

LEARNING_RATE = 3e-05
MAX_GRAD_NORM = 10

#Import model 
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name, num_labels= len(label2id))


data_explode = data['word_labels'].str.split(',').explode()
data_explode = data_explode.str.strip()
label_counts = data_explode.value_counts()
label_counts


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()
    
    for word, label in zip(sentence.split(), text_labels.split(",")):
    
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
    
        if n_subwords > 0:
            labels.append(label)
            # Extend the labels list by repeating the label `n_subwords` times
            labels.extend(['X'] * (n_subwords - 1))

    return tokenized_sentence, labels


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        item = self.data.iloc[index]
        #tokenize (and adapt corresponding labels)
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        #add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"]
        labels = ["[CLS]"] + labels + ["[SEP]"]  
        
        label_ids = [label2id[label] for label in labels]
        #truncating/padding
        maxlen = self.max_len
    
        if len(tokenized_sentence) > maxlen:
            tokenized_sentence = tokenized_sentence[:maxlen]
            label_ids = label_ids[:maxlen]
        else:
            padding_length = maxlen - len(tokenized_sentence)
            tokenized_sentence += ['[PAD]'] * padding_length
            label_ids += [-100] * padding_length  # Pad labels with -100 to ignore them
        
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
       
        return {
            'report_numbers': item['Filename'],
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


train_size = 0.8
train_dataset = data.sample(frac=train_size, random_state = 1294)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)


#display examples
for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["ids"][:30]), training_set[0]["targets"][:30]):
    print(token,label)


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

train_loader = DataLoader(training_set, **train_params)
val_loader = DataLoader(testing_set, **test_params)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

total_steps = len(train_loader) * EPOCHS  
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# Create the criterion with weights
criterion = CrossEntropyLoss(ignore_index=-100)
device = torch.device("cuda")
model = model.to(device)


# Training function with Layer-wise Learning Rate Decay
def train_epoch(model, data_loader, optimizer, scheduler, device, criterion, llrd=False):
    model.train()
    
    losses = []
    
    for d in data_loader:
        report_numbers = d['report_numbers'].to(device)
        input_ids = d['ids'].to(device)
        attention_mask = d['mask'].to(device)
        labels = d['targets'].to(device)
       # print(f'Input IDs shape: {input_ids.shape}')      
       # print(f'Attention Mask shape: {attention_mask.shape}')  
       # print(f'Labels shape: {labels.shape}') 

       
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
       # print(logits.shape)

        # Compute the loss using the criterion with weights
        #loss = criterion(logits.view(-1, 3), labels.view(-1))
        #loss = criterion(active_logits, active_labels)
        loss = criterion(logits.view(-1, 6), labels.view(-1))
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return np.mean(losses)


def seqeval_model(model, data_loader, device):
    model.eval()
    losses = []
    all_labels = []
    all_predictions = []
    
    seq_labels = []
    seq_predictions = []
    
    count = 0
    results = []
    #val_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['ids'].to(device)
            attention_mask = d['mask'].to(device)
            labels = d['targets'].to(device)
            report_numbers = d['report_numbers'] 
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits.view(-1, 6), labels.view(-1))
            losses.append(loss.item())
            
            predictions = torch.argmax(logits, dim=-1)
            
            active_labels = labels.view(-1) != -100
            active_predictions = predictions.view(-1)[active_labels]
            active_labels = labels.view(-1)[active_labels]
            
            #all_labels.extend(active_labels.cpu().numpy())
            #all_predictions.extend(active_predictions.cpu().numpy())
    
            batch_labels = active_labels.cpu().numpy()
            batch_predictions = active_predictions.cpu().numpy()
            
            seq_labels.append(batch_labels)
            seq_predictions.append(batch_predictions)
            
            results.append({
                'report_number': report_numbers.cpu().numpy(),
                'labels': active_labels.cpu().numpy(),
                'predictions': active_predictions.cpu().numpy()
            })
      
            count += 1 

    #return np.mean(losses), all_labels, all_predictions, seq_labels, seq_predictions, count
    return np.mean(losses),  seq_labels, seq_predictions, count, results



def entity_result(results):
    all_label = []
    all_pred = [] 
    for result in results:
        ids = result['labels']
        pred_ids = result['predictions']
        labels = []
        pred_labels =[]
        for tag in ids:
                #if pred_tag == 0:
                    #continue
            curr_label = id2label[tag]
            if curr_label == '[CLS]':
                continue
            elif curr_label == '[SEP]':
                break
            elif curr_label == 'X':
                continue
            labels.append(curr_label)
        for pred_tag in pred_ids:
                #if pred_tag == 0:
                    #continue
            curr_label = id2label[pred_tag]
            if curr_label == '[CLS]':
                continue
            elif curr_label == '[SEP]':
                break
            elif curr_label == 'X':
                continue
            pred_labels.append(curr_label)

        if len(pred_labels) > len(labels):
            pred_labels = pred_labels[:len(labels)]
        elif len(pred_labels) < len(labels):
            pred_labels += ['O'] * (len(labels) - len(pred_labels))

        if len(pred_labels) > len(labels):
            pred_labels = pred_labels[:len(labels)]
        elif len(pred_labels) < len(labels):
            pred_labels += ['O'] * (len(labels) - len(pred_labels))

        all_label.append(labels)
        all_pred.append(pred_labels)
    #print(classification_report(all_label, all_pred, mode='strict', scheme=IOB2))
    #print(precision_score(all_label, all_pred, mode='strict', scheme=IOB2))
    prec = round(precision_score(all_label, all_pred, mode='strict', scheme=IOB2),2)
    recall = round(recall_score(all_label, all_pred, mode='strict', scheme=IOB2),2)
    f1 = round(f1_score(all_label, all_pred, mode='strict', scheme=IOB2),2)
    return prec, recall, f1

def token_result(results):
    aggregated_label = []
    aggregated_pred = [] 
    for result in results:
        ids = result['labels']
        pred_ids = result['predictions']

        labels = []
        pred_labels =[]
        for tag in ids:

            curr_label = id2label[tag]
            if curr_label == '[CLS]':
                continue
            elif curr_label == '[SEP]':
                break
            elif curr_label == 'X':
                continue
            labels.append(curr_label)

        for pred_tag in pred_ids:

            curr_label = id2label[pred_tag]
            if curr_label == '[CLS]':
                continue
            elif curr_label == '[SEP]':
                break
            elif curr_label == 'X':
                continue
            pred_labels.append(curr_label)

        if len(pred_labels) > len(labels):
            pred_labels = pred_labels[:len(labels)]
        elif len(pred_labels) < len(labels):
            pred_labels += ['O'] * (len(labels) - len(pred_labels))

        for l in labels:
            aggregated_label.append(l)
        for p in pred_labels:
            aggregated_pred.append(p)
    
    report = clr(aggregated_label, aggregated_pred, zero_division=0, output_dict=True)
    avg_p = round((report['B-Outcome']['precision'] * report['B-Outcome']['support'] +
             report['I-Outcome']['precision'] * report['I-Outcome']['support'])/(report['B-Outcome']['support'] + report['I-Outcome']['support']),2)
    avg_r = round((report['B-Outcome']['recall'] * report['B-Outcome']['support'] +
             report['I-Outcome']['recall'] * report['I-Outcome']['support'])/(report['B-Outcome']['support'] + report['I-Outcome']['support']),2)
    avg_f = round((report['B-Outcome']['f1-score'] * report['B-Outcome']['support'] +
             report['I-Outcome']['f1-score'] * report['I-Outcome']['support'])/(report['B-Outcome']['support'] + report['I-Outcome']['support']),2)

    return avg_p, avg_r, avg_f


def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

lowest_val_loss = float('inf')
best_model = None

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')

    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, llrd=True)
    print(f'Train loss {train_loss}')
    
    model_save_path = f'./{train_name}'
    path_exist(model_save_path)
    torch.save(model.state_dict(), f'./{model_save_path}/model_{epoch + 1}.pth')
    
    val_loss, ids, pred_ids,count,results = seqeval_model(model, val_loader, device)  
    print(f'Validation loss {val_loss}')
    
    
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        
        best_model = model
        best_model_path = f'./best_model/{train_name}'
        path_exist(best_model_path)
        torch.save(model.state_dict(), f'./{best_model_path}/{train_name}_bestmodel.pth')
        print(f"Saved new best model with validation loss: {val_loss}")


def verfication(results):
    tok_p, tok_r, tok_f = token_result(results)
    en_p, en_r, en_f = entity_result(results)
    print(tok_p, tok_r, tok_f)
    print(en_p, en_r, en_f)


# +
def calculate_95_ci(data):
    mean = sum(data) / len(data)
    ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
   
    margin = (ci[1] - ci[0]) / 2
    mean = round(mean, 3)
    ci = (round(ci[0], 3), round(ci[1], 3))
    margin = round(margin, 3)
    return mean, ci
    #return f"{mean}±{margin}"

def bootstrap(results):
    all_report_numbers = []
    for result in results:
        all_report_numbers.extend(result['report_number'])
    unique_report_numbers = list(set(all_report_numbers))
    unique_count = len(unique_report_numbers)

    random.seed(42)
    tok_p_results, tok_r_results, tok_f_results = [], [], []
    en_p_results, en_r_results, en_f_results = [], [], []
    for _ in range(100):
        report_names = random.choices(unique_report_numbers, k=unique_count)
        new_results = []
        for report in report_names:
            filtered_results = [result for result in results if report == result['report_number']]
            new_results.extend(filtered_results)

        tok_p, tok_r, tok_f = token_result(new_results)
        en_p, en_r, en_f = entity_result(new_results)


        tok_p_results.append(tok_p)
        tok_r_results.append(tok_r)
        tok_f_results.append(tok_f)
        en_p_results.append(en_p)
        en_r_results.append(en_r)
        en_f_results.append(en_f)
        
    print("Token Precision 95% CI:", calculate_95_ci(tok_p_results))
    print("Token Recall 95% CI:", calculate_95_ci(tok_r_results))
    print("Token F-score 95% CI:", calculate_95_ci(tok_f_results))
    print("Entity Precision 95% CI:", calculate_95_ci(en_p_results))
    print("Entity Recall 95% CI:", calculate_95_ci(en_r_results))
    print("Entity F-score 95% CI:", calculate_95_ci(en_f_results))
    return en_p_results, en_r_results,en_f_results
# -

import json
def save_results_to_json(results, filename):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    results = convert_numpy(results)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


test = pd.read_csv("#path/to/test_data1")
test = test.reset_index(drop=True)
test_set = dataset(test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_set, **test_params)
model_path = f'./best_model/{train_name}/{train_name}_bestmodel.pth'
model.load_state_dict(torch.load(model_path))

val_loss, ids, pred_ids,count,results = seqeval_model(model, test_loader, device)  
print(f'Validation loss {val_loss}')
save_results_to_json(results, f'./{train_name}/test_data1.json')
verfication(results)
bootstrap(results)


test = pd.read_csv("#path/to/test_data2")
test = test.reset_index(drop=True)
test_set = dataset(test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_set, **test_params)
model_path = f'./best_model/{train_name}/{train_name}_bestmodel.pth'
model.load_state_dict(torch.load(model_path))

val_loss, ids, pred_ids,count,results = seqeval_model(model, test_loader, device)  
print(f'Validation loss {val_loss}')
save_results_to_json(results, f'./{train_name}/test_data2.json')
verfication(results)
bootstrap(results)


test = pd.read_csv("#path/to/test_data3")
test = test.reset_index(drop=True)
test_set = dataset(test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_set, **test_params)
model_path = f'./best_model/{train_name}/{train_name}_bestmodel.pth'
model.load_state_dict(torch.load(model_path))

val_loss, ids, pred_ids,count,results = seqeval_model(model, test_loader, device)  
print(f'Validation loss {val_loss}')
save_results_to_json(results, f'./{train_name}/test_data3.json')
verfication(results)
bootstrap(results)


test = pd.read_csv("#path/to/test_data4")
test = test.reset_index(drop=True)
test_set = dataset(test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_set, **test_params)
model_path = f'./best_model/{train_name}/{train_name}_bestmodel.pth'
model.load_state_dict(torch.load(model_path))

val_loss, ids, pred_ids,count,results = seqeval_model(model, test_loader, device)  
print(f'Validation loss {val_loss}')
save_results_to_json(results, f'./{train_name}/test_data4')
verfication(results)
bootstrap(results)



