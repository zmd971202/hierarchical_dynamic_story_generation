import torch
import torch.nn as nn
import torch.nn.functional as F

class Updater(nn.Module):
    
    def __init__(self, hps):
        super(Updater, self).__init__()
        
        sent_hidden_size = hps.sent_hidden_size
        word_hidden_size = hps.word_hidden_size
        char_emb_size = hps.char_emb_size
        dropout = hps.dropout

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sent_hidden_size + word_hidden_size, char_emb_size)

    
    def forward(self, alignments, hidden):
        '''
        alignments: [batch, num_char]
        hidden: [batch, sent_hidden_size + word_hidden_size]
        '''
        hidden = self.dropout(hidden) #[batch, sent_hidden_size + word_hidden_size]
        hidden = self.fc(hidden) #[batch, char_emb_size]
        hidden = hidden.unsqueeze(1) #[batch, 1, char_emb_size]
        alignments = alignments.unsqueeze(2) #[batch, num_char, 1]
        update = alignments.mul(hidden) #[batch, num_char, char_emb_size]
        
        return update #[batch, num_char, char_emb_size]