import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapted_GRU(nn.Module):
    
    def __init__(self, hps):
        super(Adapted_GRU, self).__init__()
        
        word_emb_size = hps.word_emb_size
        char_emb_size = hps.char_emb_size
        word_hidden_size = hps.word_hidden_size
        sent_hidden_size = hps.sent_hidden_size
        dropout = hps.dropout
#         assert word_emb_size == char_emb_size and word_emb_size == sent_hidden_size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.s2h = nn.Linear(word_hidden_size, 5 * sent_hidden_size)
        self.c2h = nn.Linear(char_emb_size, 5 * sent_hidden_size)
        self.a2h = nn.Linear(word_emb_size, 5 * sent_hidden_size)
        self.h2h = nn.Linear(sent_hidden_size, 5 * sent_hidden_size)
        self.t2h = nn.Linear(word_emb_size, sent_hidden_size)
        self.r2h = nn.Linear(char_emb_size, sent_hidden_size)
        
    
    def forward(self, title, char_emb, action_emb, R, sent_h, prev_hidden):
        '''
        title: [batch, word_emb_size]
        char_emb: [batch, char_emb_size]
        action_emb: [batch, verb_emb_size]
        R: [batch, num_char, char_emb_size]
        sent_h: [batch, word_hidden_size]
        prev_hidden: [batch, sent_hidden_size]
        '''
        
#         char = self.c2h(self.dropout1(char_emb)) #[batch, 5 * sent_hidden_size]
#         action = self.a2h(self.dropout2(action_emb))
#         hidden = self.h2h(self.dropout3(prev_hidden))
        
#         c1, c2, c3, c4, c5 = char.chunk(5, 1)
#         a1, a2, a3, a4, a5 = action.chunk(5, 1)
#         h1, h2, h3, h4, h5 = hidden.chunk(5, 1)
        
#         rt = torch.sigmoid(c2 + a2 + h2) #[batch, sent_hidden_size]
#         st = torch.sigmoid(c3 + a3 + h3)
#         zt = torch.sigmoid(c4 + a4 + h4)
#         qt = torch.sigmoid(c5 + a5 + h5)
        
#         title = self.t2h(self.dropout4(title)) #[batch, sent_hidden_size]
#         r = self.r2h(self.dropout5(R)) #[batch, num_char, sent_hidden_size]
#         r = torch.sum(r, dim=1) #[batch, sent_hidden_size]
        
#         new_hidden = torch.tanh(c1 + a1 + rt * h1 + st * title + qt * r) #[batch, sent_hidden_size]
        
#         new_hidden = zt * prev_hidden + (1 - zt) * new_hidden #[batch, sent_hidden_size]
        
#         return new_hidden
        
        char = self.c2h(self.dropout1(char_emb)) #[batch, 5 * sent_hidden_size]
        action = self.a2h(self.dropout2(action_emb))
        hidden = self.h2h(self.dropout3(prev_hidden))
        sent = self.s2h(self.dropout6(sent_h))
        
        c1, c2, c3, c4, c5 = char.chunk(5, 1)
        a1, a2, a3, a4, a5 = action.chunk(5, 1)
        h1, h2, h3, h4, h5 = hidden.chunk(5, 1)
        s1, s2, s3, s4, s5 = sent.chunk(5, 1)
        
        rt = torch.sigmoid(c2 + a2 + h2 + s2) #[batch, sent_hidden_size]
        st = torch.sigmoid(c3 + a3 + h3 + s3)
        zt = torch.sigmoid(c4 + a4 + h4 + s4)
        qt = torch.sigmoid(c5 + a5 + h5 + s5)
        
        title = self.t2h(self.dropout4(title)) #[batch, sent_hidden_size]
        r = self.r2h(self.dropout5(R)) #[batch, num_char, sent_hidden_size]
        r = torch.sum(r, dim=1) #[batch, sent_hidden_size]
        
        new_hidden = torch.tanh(c1 + a1 + s1 + rt * h1 + st * title) #[batch, sent_hidden_size]
        
        new_hidden = zt * prev_hidden + (1 - zt) * new_hidden #[batch, sent_hidden_size]
        
        return new_hidden