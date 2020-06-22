import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Character_Embedding_Initializer(nn.Module):
    
    def __init__(self, hps):
        super(Character_Embedding_Initializer, self).__init__()
        self.char_emb_size = hps.char_emb_size
        self.char_hidden_size = hps.char_hidden_size
        self.max_num_char = hps.max_num_char
        word_emb_size = hps.word_emb_size
        dropout = hps.dropout
        
        self.gru = nn.GRU(word_emb_size, self.char_hidden_size, 1, batch_first=True, bidirectional=True)
#         self.lstm = nn.LSTM(word_emb_size, self.char_hidden_size, 1, batch_first=True, bidirectional=True)
        self.compat = nn.Linear(2 * self.char_hidden_size, self.char_emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, titles, attributes, attr_lens):
        '''
        titles: [batch, title_len + 1, word_emb_size]
        attributes: [num_char, batch, num_attr, word_emb_size]
        attr_lens: [num_char, batch]
        #all_chars: [batch, num_char, char_len, word_emb_size]
        '''
        
#         R = torch.mean(attributes, dim=2) #[batch, num_char, word_emb_size]
#         return R
        
#         batch_size, num_char, _, word_emb_size = all_chars.size()
        batch_size, _, word_emb_size = attributes[0].size()
#         title_len = titles.size(1) - 1
#         R = torch.FloatTensor(batch_size, num_char, self.char_emb_size).cuda()
        
#         titles = titles.unsqueeze(1) #[batch, 1, title_len + 1, word_emb_size]
#         titles = titles.expand(batch_size, num_char, title_len + 1, word_emb_size) #[batch, num_char, title_len + 1, word_emb_size]
#         cat = torch.cat((titles, attributes), 2)  #[batch, num_char, title_len + 1 + num_attr, word_emb_size]
#         cat = torch.cat((titles, all_chars), 2)  #[batch, num_char, title_len + 1 + char_len, word_emb_size]
#         cat = all_chars

        
        hs = []
        for i in range(self.max_num_char):
            attr = pack_padded_sequence(attributes[i], attr_lens[i], batch_first=True, enforce_sorted=False)
            
            h = self.init_hidden(batch_size)
            _, h = self.gru(attr, h) #[2, batch, char_hidden_size]

#             h, cell = self.init_hidden(batch_size)
#             _, (h, cell) = self.lstm(attr, (h, cell)) #[2, batch, char_hidden_size]
        
        
            h = h.view(batch_size, -1)
            h = self.dropout(h) #[batch, char_hidden_size]
            h = self.compat(h) #[batch, char_emb_size]
            hs.append(h)

        R = torch.stack(hs, 1) #[batch, num_char, char_emb_size]

        return R
    
    def init_hidden(self, batch_size):
        weight = next(self.gru.parameters()).data
#         hidden = weight.new(1, batch_size, self.char_hidden_size).zero_().cuda()
        hidden = nn.init.xavier_uniform_(weight.new(2, batch_size, self.char_hidden_size)).cuda()
        return hidden
        
#         hidden = nn.init.xavier_uniform_(next(self.lstm.parameters()).data.new(2, batch_size, self.char_hidden_size)).cuda()
#         cell =  nn.init.xavier_uniform_(next(self.lstm.parameters()).data.new(2, batch_size, self.char_hidden_size)).cuda()
#         return (hidden, cell)
        