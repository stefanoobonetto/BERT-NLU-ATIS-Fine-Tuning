import torch 
from conll import evaluate
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() 

        slots, intent = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])

        loss_intent = criterion_intents(intent, sample['intents'])

        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot 

        loss_array.append(loss.item())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() 
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 
        for sample in data:
            slots, intents = model(sample['utterances'], attentions=sample['attention_mask'], token_type_ids=sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            
            loss_slot = criterion_slots(slots, sample['y_slots'])
            
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]

            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
                


            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)

            
            for id_seq, seq in enumerate(output_slots):                         # iterate over the output slots for each sentence

                length = sample['slots_len'].tolist()[id_seq]
                
                # get the token ids of the utterance and ground truth slots
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                
                # convert ground truth slot ids to slot labels
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                
                # convert utterance token ids to tokens
                utterance = [lang.id2word[elem] for elem in utt_ids]                
                
                # get the decoded slot ids
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])

                tmp_seq = []
                
                for id_el, elem in enumerate(to_decode):                        # iterate over decoded slot ids and convert them to slot labels
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))

                hyp_slots.append(tmp_seq)

    try:            

        results = evaluate(ref_slots, hyp_slots)
    
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array
