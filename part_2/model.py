import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel #, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class JointBERT(BertPreTrainedModel):
    def __init__(self, hid_size, len_vocab_intent,  len_vocab_slot, config):
        super(JointBERT, self).__init__(config)

        print(len_vocab_slot)

        print(len_vocab_intent)        

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_layer = nn.Linear(hid_size, len_vocab_intent)
        self.slot_layer = nn.Linear(hid_size, len_vocab_slot)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids) #, attention_mask=attention_mask,
                            #token_type_ids=token_type_ids) 
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent = self.intent_layer(pooled_output)
        slots = self.slot_layer(sequence_output[:, 0, :])  # Take the [CLS] token only
        # slots = self.slot_layer(sequence_output)
        
        print(slots.shape)
        print(intent.shape)
        return slots, intent  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
