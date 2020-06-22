import torch
import torch.nn as nn
import torch.nn.functional as F

class Action_Selector(nn.Module):
    
    def __init__(self, hps):
        super(Action_Selector, self).__init__()
        
        sent_hidden_size = hps.sent_hidden_size
        word_emb_size = hps.word_emb_size
        vocab_size = hps.vocab_size
        dropout = hps.dropout
        
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
#         self.layer1 = nn.Linear(sent_hidden_size + char_emb_size, 1024)
#         self.layer2 = nn.Linear(512, 1024)
        self.layer3 = nn.Linear(sent_hidden_size + word_emb_size, vocab_size)
    
    def forward(self, hidden, char_emb):
        '''
        hidden: [batch, 2 * hidden_size]
        char_emb: [batch, word_emb_size]
        '''
#         hidden = hidden.squeeze(0) #[batch, sent_hidden_size]
        cat = torch.cat((hidden, char_emb), 1) #[batch, 2 * hidden_size + word_emb_size]
#         cat = hidden
#         cat = self.layer1(self.dropout1(cat))
#         cat = self.layer2(self.dropout2(cat))
        cat = self.layer3(self.dropout3(cat)) #[batch, vocab_size]
        
        distributions = F.log_softmax(cat, dim=-1)
        id = torch.argmax(distributions, -1)
        return cat, id #[batch, vocab_size], [batch]