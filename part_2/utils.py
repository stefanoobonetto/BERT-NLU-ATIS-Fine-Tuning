# Global variables
import os
import torch
import json
from pprint import pprint
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn as nn
import torch.utils.data as data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
PAD_TOKEN = 0

def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):                    # creates a dictionary that maps unique id to each word
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):                               # creates a dictionary that maps unique id to each label    
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
            vocab['unk'] = len(vocab)
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.changed = [False for _ in range(len(dataset))] # To keep track of changed slots
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def convert_to_bert_dataset(self, lang, tokenizer, data_raw):

        bert_utt = []



        for string in data_raw:
            bert_utt.append(tokenizer(string['utterance'], return_tensors='pt'))
        
        # print("bert_utt: ", bert_utt)

        # print(train_raw[0])
                                        # {'intent': 'flight',
                                        # 'slots': 'O O O B-airline_name O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name',
                                        # 'utterance': 'is there a delta flight from denver to san francisco'}]

        # print(bert_train_utt[0])         
                                        # {'input_ids': tensor([[ 101, 2003, 2045, 1037, 7160, 3462, 2013, 7573, 2000, 2624, 3799,  102]]), 
                                        # 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
                                        # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        # print(len(train_raw[0]['utterance'].split()))

        # print(len(bert_train_utt[0]['input_ids'][0]))


        # train_set_bert = [{'intent': y, 'slots': '', 'utterance': x['input_ids']} for x, y in zip(bert_utt, self.intent_ids)]
        
        # print(len(train_set_bert[0]['utterance']))

        # train_set_bert_changed = [{'intent': '', 'slots': '', 'utterance': x['input_ids']} for x in bert_train_utt]

        new_slots = []
        for id, item in enumerate(data_raw):
            new_slot = "O "
            # print("item: ", item['utterance'], ", n: ", len(item['utterance'].split()))
            # print("------> utt_id[0]", self.utt_ids[id])

            # print("len_item: ", len(item['utterance'].split()), ", len_utt_ids: ", len(bert_utt[id]['input_ids'][0]) - 2)

            if len(item['utterance'].split()) == len(bert_utt[id]['input_ids'][0]) - 2:               # no variation in splitting
                new_slot += item['slots'] 
                new_slot += " O"

                # print("[NO VAR]\n", item['slots'], "\ninto\n", new_slot, "\n\n")

                # print("----------> no variation in splitting", id)
            else:
                # print("variation in splitting", id)
                self.changed[id] = True
                slots_list = item['slots'].split()
                # print("[VAR]\n", item['slots'], "")
                # print("slots_list: ", slots_list, ", len: ", len(slots_list))
                # print("item: ", item['utterance'], ", n: ", len(item['utterance'].split()))
                for i, word in enumerate(item['utterance'].split()):
                    bert_word = tokenizer(word)['input_ids']
                    # print("bert_word: ", bert_word, ", len: ", len(bert_word))
                    new_slot += slots_list[i] + " "
                    for j in range(len(bert_word) - 3):
                        # print("before: ", new_slot)
                        new_slot += "O "
                        # print("after: ", new_slot)
                        # print(word, " splitted in two slots, adding a 0 after ", slots_list[i])
                new_slot += "O"
                # print("into\n", new_slot, "\n\n")
            new_slots.append(new_slot)
            # print("\n")

        self.change_slot_ids(new_slots, lang)

        # for id, item in enumerate(data_raw):
        #     print("\n\nitem_normal: \n", item["utterance"], "\n", item["slots"], "\n", item["intent"])
        #     print("\nitem_bert: \n", self.utt_ids[id], "\n", self.slots[id], " ---> ", self.slot_ids[id], "\n", self.intents[id], " ---> ",self.intent_ids[id], "\n changed: ", self.changed[id])

        return "--------> OK FINITOOOOO"


    def change_slot_ids(self, new_slots, lang):
        self.slots = new_slots

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)

    def change_intent_ids(self, new_intent, lang):
        self.intents = new_intent
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    # tmp_seq.append(mapper[self.unk])
                    tmp_seq.append(mapper['unk'])
            res.append(tmp_seq)
        return res
    
        
def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

