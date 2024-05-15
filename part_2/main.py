import torch.optim as optim
from utils import *
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import JointBERT
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertConfig
from pprint import pprint
from function import eval_loop, train_loop, init_weights, save_results

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

hid_size = 768
emb_size = 300  

lr = 5                                   # learning rate
clip = 5                                 # Clip the gradients

tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

# pprint(tmp_train_raw)
portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stringatify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])

# Random stringatify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff

corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                        # however this depends on the research purpose

slots = set(sum([line['slots'].split() for line in corpus],[]))

intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)

out_slot = len(lang.slot2id)
out_int = len(lang.intent2id)
vocab_len = len(lang.word2id)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
# model = BertModel.from_pretrained("bert-base-uncased") # Download the model


train_dataset = IntentsAndSlots(train_raw, lang)
# print(train_dataset.mapping_seq(train_dataset.utterances, lang.word2id))

dev_dataset = IntentsAndSlots(dev_raw, lang)
test_dataset = IntentsAndSlots(test_raw, lang)

train_dataset.convert_to_bert_dataset(lang, tokenizer, train_raw)
dev_dataset.convert_to_bert_dataset(lang, tokenizer, dev_raw)
test_dataset.convert_to_bert_dataset(lang, tokenizer, test_raw)

config = BertConfig.from_pretrained('bert-base-uncased')  # Load BERT configuration

vocab_len_slots = len(lang.slot2id)
print("Vocabulary length of slots:", vocab_len_slots)

model = JointBERT(hid_size, len(lang.intent2id), len(lang.slot2id), config).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
criterion_intents = nn.CrossEntropyLoss()                                   # Because we do not have the pad token

print("TUTTO OK FINO A QUI")

# Dataloader instantiations
train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

n_epochs = 200
patience = 3
losses_train = []
losses_dev = []
sampled_epochs = []
best_f1 = 0
for x in tqdm(range(1,n_epochs)):
    loss = train_loop(train_loader, optimizer, criterion_slots, 
                      criterion_intents, model, clip=clip)
    if x % 5 == 0: # We check the performance every 5 epochs
        sampled_epochs.append(x)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                      criterion_intents, model, lang)
        losses_dev.append(np.asarray(loss_dev).mean())
        
        f1 = results_dev['total']['f']
        # For decreasing the patience you can also use the average between slot f1 and intent accuracy
        if f1 > best_f1:
            best_f1 = f1
            # Here you should save the model
            patience = 3
        else:
            patience -= 1
        if patience <= 0: # Early stopping with patience
            break # Not nice but it keeps the code clean

results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                         criterion_intents, model, lang)    

print('Slot F1: ', results_test['total']['f'])
print('Intent Accuracy:', intent_test['accuracy'])

save_results(lr, x, sampled_epochs, losses_dev, losses_train)

plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
plt.title('Train and Dev Losses')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(sampled_epochs, losses_train, label='Train loss')
plt.plot(sampled_epochs, losses_dev, label='Dev loss')
plt.legend()
plt.show()



















# # This Python script is implementing a training pipeline for a model that performs intent classification 
# # and slot filling on the ATIS (Airline Travel Information Systems) dataset. Here's a step-by-step breakdown:
# # - Imports and setup: The script imports necessary libraries and sets up some parameters like hidden 
# #   layer size (hid_size), embedding size (emb_size), learning rate (lr), and gradient clipping value (clip).
# # - Data loading and preprocessing: The script loads the ATIS training and test datasets. It then stringatifies 
# #   the training data based on the intent labels, ensuring that intents occurring only once are put in the training 
# #   set. The training data is then split into a training set and a development set.
# # - Language processing: The script processes the words, slots, and intents in the corpus to create a Lang object, 
# #   which is used for converting between words/intents/slots and their corresponding IDs.
# # - Model setup: The script initializes a ModelIAS model with the specified parameters and applies weight initialization. 
# #   It also sets up the optimizer and loss functions.
# # - Data loading for training: The script creates PyTorch DataLoader objects for the training, development, and test datasets.
# # - Training loop: The script trains the model for a specified number of epochs (n_epochs). Every 5 epochs, it evaluates 
# #   the model on the development set and computes the F1 score. If the F1 score does not improve after a certain number of 
# #   epochs (patience), the training is stopped early.
# # - Evaluation: After training, the script evaluates the model on the test set and prints out the F1 score for slot filling
# #   and the accuracy for intent classification.
# # - Plotting: Finally, the script plots the training and development losses over the epochs.

# # The ModelIAS class has the following components:
# # - Embedding layer: This layer converts the input words (represented as integers) into dense vectors of fixed size.
# # - LSTM layer (utt_encoder): This layer encodes the input sequence into a fixed-length vector. It's a unidirectional LSTM 
# #   with a specified number of layers.
# # - Linear layers (slot_out and intent_out): These layers map the LSTM output to the slot and intent logits, respectively.
# # - Dropout layer: This layer randomly sets a fraction of input units to 0 at each update during training time, which helps 
# #   prevent overfitting. However, it's not currently used in the forward method.

# # The forward method of ModelIAS processes the input utterance and sequence lengths as follows:
# # - The utterance is passed through the embedding layer to get the word embeddings.
# # - The embeddings are packed into a PackedSequence object using pack_padded_sequence to avoid unnecessary computations 
# #   on padding tokens.
# # - The packed sequence is passed through the LSTM layer to get the encoded sequence and the last hidden state.
# # - The encoded sequence is unpacked using pad_packed_sequence.
# # - The unpacked sequence and the last hidden state are passed through the slot and intent output layers, respectively, to 
# #   get the slot and intent logits.
# # - The slot logits are permuted for loss computation.

# # The init_weights function initializes the weights of the LSTM and Linear layers in a specific way: LSTM weights are initialized 
# # with Xavier uniform or orthogonal initialization, and LSTM biases are set to 0. Linear weights are initialized with uniform 
# # stringibution between -0.01 and 0.01, and biases are set to 0.01.