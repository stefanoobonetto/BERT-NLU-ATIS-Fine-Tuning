from function import *
from utils import *
from model import ModelIAS

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import os 
from tqdm import tqdm
import numpy as np

hid_size = 300
emb_size = 300

# the below flags can be used to run the script with different configurations

DROP = True                                     # dropout layer
BIDIRECTIONAL = True                            # bidirectional LSTM

PAD_TOKEN = 0

lr = 0.00001                             # learning rate
clip = 5                                # Clip the gradient

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

portion = 0.10

intents = [x['intent'] for x in tmp_train_raw]          # stringatify on intents
count_y = Counter(intents)                              # count the number of occurences of each intent

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # if some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])

# random stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

words = sum([x['utterance'].split() for x in train_raw], []) 

corpus = train_raw + dev_raw + test_raw     # don't want unk labels

slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, drop=DROP, bidirectional=BIDIRECTIONAL).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()                                       # we don't have the pad token

# Create our datasets
train_dataset = IntentsAndSlots(train_raw, lang)
dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

n_epochs = 200
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = -1

for x in tqdm(range(1,n_epochs)):
    loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
    sampled_epochs.append(x)
    losses_train.append(np.asarray(loss).mean())
    results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)

    losses_dev.append(np.asarray(loss_dev).mean())

    f1 = results_dev['total']['f']

    if f1 > best_f1:
        best_f1 = f1
        patience = 3
    else:
        patience -= 1
    if patience <= 0:
        break

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
    
print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])
