import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel #, BertConfig

class JointBERT(BertPreTrainedModel):
    def __init__(self, hid_size, len_vocab_intent, len_vocab_slot, config):
        super(JointBERT, self).__init__(config)

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_layer = nn.Linear(hid_size, len_vocab_intent)
        self.slot_layer = nn.Linear(hid_size, len_vocab_slot)

    def forward(self, input_ids):

        utt_encoded = self.bert(input_ids)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(utt_encoded)

        return slots, intent  