import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel #, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class JointBERT(BertPreTrainedModel):
    def __init__(self, hid_size, len_vocab_intent, len_vocab_slot, config):
        super(JointBERT, self).__init__(config)

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_layer = nn.Linear(hid_size, len_vocab_intent)
        self.slot_layer = nn.Linear(hid_size, len_vocab_slot)

    def forward(self, input_ids):

        utt_encoded = self.bert(input_ids)

        # Compute slot logits
        slots = self.slot_layer(utt_encoded.last_hidden_state)
        # Compute intent logits
        intent = self.intent_layer(utt_encoded.last_hidden_state[:,0,:])

        return slots, intent  
    

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
