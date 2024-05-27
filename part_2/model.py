import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel #, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class JointBERT(BertPreTrainedModel):
    def __init__(self, hid_size, len_vocab_intent,  len_vocab_slot, config):
        super(JointBERT, self).__init__(config)     

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_layer = nn.Linear(hid_size, len_vocab_intent)
        self.slot_layer = nn.Linear(hid_size, len_vocab_slot)

    def forward(self, utterance, seq_lengths, attention_mask=None):
        # utterance.size() = batch_size X seq_len
        
        # Get BERT embeddings
        bert_emb = self.bert(utterance)[0] # bert_emb.size() = batch_size X seq_len X hid_size
        
        # Compute slot logits
        slots = self.slot_layer(bert_emb)
        # Compute intent logits
        intent = self.intent_layer(bert_emb[:, 0, :]) # Use the first token's representation
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0, 2, 1) # We need this for computing the loss

        # Slot size: batch_size, classes, seq_len
        return slots, intent