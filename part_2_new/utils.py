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

# device = 'cpu'

from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertConfig
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer

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
    def __init__(self, intents, slots, cutoff=0):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):                    # creates a dictionary that maps unique id to each word
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = -1
        
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
        self.slot_ids = []
        self.intent_ids = []
        self.utt_ids = []

        
        self.unk = unk
        
        for x in dataset:
            self.utterances.append("[CLS] " + x['utterance'] + " [SEP]")
            self.slots.append("O " + x['slots'] + " O")
            self.intents.append(x['intent'])

        utter_ids, res_slot, res_attention, res_token_type_id = self.mapping_seq(self.utterances, self.slots, tokenizer, lang.slot2id)

        for utt, att, token in zip(utter_ids, res_attention, res_token_type_id):
            self.utt_ids.append({'input_ids': utt, 'attention_mask': att, 'token_type_ids': token})

        for elem in res_slot:
            self.slot_ids.append({'input_ids': elem, 'attention_mask': [], 'token_type_ids': []})

        intent_ids = self.mapping_lab(self.intents, lang.intent2id)
        for elem in intent_ids:
            self.intent_ids.append(elem)

        # for intent, intent_id in zip(self.intents, self.intent_ids):
        #     print("Intent: ", str(intent)) 
        #     print("Intent[IDs]: ", str(intent_id))
        #     print("Translated id to token: ", lang.id2intent[intent_id])

        # for i in range(len(self.utterances)):
        #     print("Phrase:                      ", self.utterances[i])
        #     print("Phrase[IDs]:                 ", self.utt_ids[i]['input_ids'])

        #     # print("last_id: ", self.utt_ids[i]['input_ids'][-1], " correspond to ", tokenizer.convert_ids_to_tokens([self.utt_ids[i]['input_ids'][-1]]))
        #     print("Slots:                       ", self.slots[i])
        #     print("Slots[IDs]:                  ", self.slot_ids[i]['input_ids'])
        #     print("Intent:                      ", self.intents[i])
        #     print("Intent[IDs]:                 ", self.intent_ids[i]['input_ids'])

        #     print("\n\n")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx]["input_ids"])
        slots = torch.Tensor(self.slot_ids[idx]["input_ids"])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    def mapping_lab(self, data, mapper):
        # res = []
        # for seq in data:
            #res.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq)))
            # print("\n", seq.split(), "\n", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(seq)), "\n\n")
        # return res
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, utterances, slots, tokenizer, mapper_slot): # Map sequences to number
        res_utt = []
        res_slot = []
        res_attention = []
        res_token_type_id = []
        
        for sequence, slot in zip(utterances, slots):
            # print("utterance: ", sequence, ",            len: ", len(sequence.split()), "len_bert: ", len(tokenizer(sequence)['input_ids']))
            tmp_seq = []
            tmp_slot = []
            tmp_attention = []
            tmp_token_type_id = []

            for word, elem in zip(sequence.split(), slot.split(' ')):
                tmp_attention.append(1)
                tmp_token_type_id.append(0)
                word_tokens = tokenizer(word)
                word_tokens = word_tokens[1:-1]

                tmp_seq.extend(word_tokens['input_ids'])
                tmp_slot.extend([mapper_slot[elem]] + [mapper_slot['pad']] * (len(word_tokens['input_ids']) - 1))

                for i in range(len(word_tokens['input_ids']) - 1):
                    tmp_attention.append(0)
                    tmp_token_type_id.append(0)

            res_utt.append(tmp_seq)
            res_slot.append(tmp_slot)
            res_attention.append(tmp_attention)
            res_token_type_id.append(tmp_token_type_id)

            
            # print("utterance: ", tokenizer.tokenize(sequence), ",            len: ", len(tokenizer.tokenize(sequence)), "\nutterance_bert: ", tokenizer.convert_ids_to_tokens(tmp_seq) ,"len_bert: ", len(tmp_seq), "\n\n")

        return res_utt, res_slot, res_attention, res_token_type_id

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

    # print(data[0])                          # {'utterance': tensor([  101., 25493.,  7599.,  2013.,  2047.,  2259.,  2103.,  2000.,  5869.,
                                            # 7136.,  2006.,  4465.,   102.]), 'slots': tensor([ 36.,   7.,  36.,  36.,   8.,  40.,  40.,  36., 110.,  20.,  36.,  59.,
                                            # 36.]), 'intent': [3462]}
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])                  # input_ids': utt, 'attention_mask': att, 'token_type_ids': token})
    
    y_slots, y_lengths = merge(new_item["slots"])
    # print("intent: ", new_item["intent"])
    intent = torch.LongTensor(new_item["intent"])
    # intent = pad_sequence(intent, batch_first=True, padding_value=0)        # Pad the sequences (some intents may be composed by more tokens)
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    tmp_att = []
    tmp_token_type_id = []
    src_utt_attention = []
    src_utt_token_type = []

    for seq in new_item["utterance"]:
        for input_id in seq:
            tmp_att.append(input_id != 0)
            tmp_token_type_id.append(0)
        src_utt_attention.append(tmp_att)
        src_utt_token_type.append(tmp_token_type_id)

    src_utt_attention = torch.tensor(src_utt_attention).to(device)
    src_utt_token_type = torch.tensor(src_utt_token_type).to(device)

    # print("attention: ", src_utt_attention)
    # print("token_type: ", src_utt_token_type)
    

    new_item["utterances"] = src_utt
    new_item["attention_mask"] = src_utt_attention
    new_item["token_type_ids"] = src_utt_token_type
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    # print("type of slots: ", type(new_item["y_slots"]))
    # print("shape of slots: ", new_item["y_slots"].shape)
    return new_item

