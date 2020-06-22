import torch
import torch.nn as nn
import torch.nn.functional as F

class Character_Selector(nn.Module):
    
    def __init__(self, hps):
        super(Character_Selector, self).__init__()
        sent_hidden_size = hps.sent_hidden_size
        char_emb_size = hps.char_emb_size
        
        self.attn = nn.Linear(hps.char_emb_size, sent_hidden_size)
        self.dropout = nn.Dropout(hps.dropout)
        self.fc = nn.Linear(char_emb_size, hps.max_num_char)
#         self.fc = nn.Linear(sent_hidden_size, hps.max_num_char)
    
    def forward(self, hidden, R, masks):
        '''
        hidden: [batch, sent_hidden_size]
        R: [batch, num_char, char_emb_size]
        masks: [batch, num_char] 0(pad)/1(char) 
        '''
#         alignments = torch.bmm(self.attn(R), hidden.permute(1, 2, 0)) #[batch, num_char, 1]
#         alignments = alignments.squeeze(2) #[batch, num_char]
#         masked_alignments = alignments + (masks + 1e-45).log()
#         alignments = F.softmax(masked_alignments, dim=-1) #[batch, num_char]
#         chars = torch.sum(R.mul(alignments.unsqueeze(-1)), dim=1)  #[batch, char_emb_size]

#         return chars, alignments, masked_alignments #[batch, char_emb_size], [batch, num_char], [batch, num_char]
        
    
        alignments = torch.bmm(self.attn(R), hidden.unsqueeze(2)) #[batch, num_char, 1]
        masked_alignments = alignments.squeeze(2) + (masks + 1e-45).log()
#         alignments = alignments.squeeze(2)
#         masked_alignments = alignments.masked_fill(~masks, torch.finfo(alignments.dtype).min)
#         print('1: ', masked_alignments)
        alignments = F.softmax(masked_alignments, dim=-1).unsqueeze(-1)
#         print('2: ', alignments)
#         alignments = alignments.squeeze(2) #[batch, num_char]
#         masked_alignments = alignments + (masks + 1e-45).log()
        chars = torch.sum(R.mul(alignments), dim=1) #[batch, char_emb_size]
        alignments = self.fc(self.dropout(chars)) #[batch, num_char]
        masked_alignments = alignments + (masks + 1e-45).log()
#         masked_alignments = alignments.masked_fill(~masks, torch.finfo(alignments.dtype).min)
        alignments = F.softmax(masked_alignments, dim=-1)
          #[batch, char_emb_size]
#         print('masked_alignments: ', masked_alignments)
        return chars, alignments, masked_alignments #[batch, char_emb_size], [batch, num_char], [batch, num_char]
    
#         self.fc(hidden)
#         return self.fc(hidden.squeeze(0)) + (masks + 1e-45).log()
        
        
        