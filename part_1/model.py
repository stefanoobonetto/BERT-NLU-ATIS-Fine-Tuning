import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, drop=False, bidirectional=False):
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.drop = drop
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)    
        self.bidirectional = bidirectional

        if bidirectional:
            hid_size *= 2                   # bidirectional --> double the hid_size for linear layers

        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)
        else:
            last_hidden = last_hidden[-1,:,:]

        if self.drop:
            utt_encoded = self.dropout(utt_encoded)
            last_hidden = self.dropout(last_hidden)

        slots = self.slot_out(utt_encoded)
        intent = self.intent_out(last_hidden)
        slots = slots.permute(0,2,1)

        return slots, intent

