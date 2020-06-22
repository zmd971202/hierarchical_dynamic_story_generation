import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from adapted_gru import Adapted_GRU
from char_emb_initializer import Character_Embedding_Initializer
from char_selector import Character_Selector
from action_selector import Action_Selector
from sent_generator import Sent_Generator
from updater import Updater

class GlobalAttention(nn.Module):
    """
    Global Attention as described in 'Effective Approaches to Attention-based Neural Machine Translation'
    """
    def __init__(self, enc_hidden, dec_hidden, dropout=0.1):
        super(GlobalAttention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        # a = h_t^T W h_s
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear_in = nn.Linear(dec_hidden, enc_hidden, bias=False)
        # W [c, h_t]
        self.linear_out = nn.Linear(dec_hidden + enc_hidden, dec_hidden)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1))).long()

    def forward(self, inputs, context, mask):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        mask: [batch, tgt_len, src_len]
        #context_lengths (LongTensor): the source context lengths.
        """
        # (batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()



#         mask = self.sequence_mask(context_lengths, src_len)
        # (batch, 1, src_len)
#         mask = mask.unsqueeze(1).cuda()  # Make it broadcastable.
#         if next(self.parameters()).is_cuda:
#             mask = mask.cuda()
#         print(align.size())
#         print(mask.size())
        align = align + (mask + 1e-45).log()
#         align.data.masked_fill_(~mask, -float('inf')) # fill <pad> with -inf

        align_vectors = self.softmax(align.view(batch*tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden) -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat((c, inputs), 2).view(batch*tgt_len, self.enc_hidden + self.dec_hidden)
        attn_h = self.tanh(self.linear_out(self.dropout2(concat_c)).view(batch, tgt_len, self.dec_hidden)) 

        # transpose will make it non-contiguous
#         attn_h = attn_h.transpose(0, 1).contiguous()
#         align_vectors = align_vectors.transpose(0, 1).contiguous()
        # (tgt_len, batch, dim)
        return attn_h, align_vectors

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()
#         print(h_t.size())
#         print(h_s.size())
        h_t = h_t.contiguous().view(tgt_batch*tgt_len, tgt_dim)
        h_t_ = self.linear_in(self.dropout1(h_t))
#         print(h_t_.size())
        h_t_ = h_t_.contiguous().view(tgt_batch, tgt_len, src_dim)
        # (batch, d, s_len)
        h_s_ = h_s.transpose(1, 2)
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t_, h_s_)

class Model(nn.Module):
    
    def __init__(self, char_weights, action_weights, hps):
        super(Model, self).__init__()
        word_emb_size = hps.word_emb_size
        self.word_hidden_size = hps.word_hidden_size
        self.vocab_size = hps.vocab_size
        self.sent_hidden_size = hps.sent_hidden_size
        self.max_sent_len = hps.max_sent_len
        self.alpha1 = hps.alpha1
        self.alpha2 = hps.alpha2
        self.alpha3 = hps.alpha3
        self.beta = hps.beta
        dropout = hps.dropout
        
        self.embedding = nn.Embedding(self.vocab_size, word_emb_size, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        self.t2h = nn.Linear(word_emb_size, self.sent_hidden_size)
        self.gru = Adapted_GRU(hps)
        
#         self.lstm = nn.GRU(word_emb_size, self.hidden_size, 1, batch_first=True)
        self.sent_generator = Sent_Generator(hps)
        
        self.char_emb_initializer = Character_Embedding_Initializer(hps)
        self.char_selector = Character_Selector(hps)
        self.action_selector = Action_Selector(hps)
        self.updater = Updater(hps)
        
        self.attn = GlobalAttention(self.sent_hidden_size, self.sent_hidden_size, dropout)
        
#         self.char_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor(char_weights))
        self.action_criterion = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor(action_weights))
        self.word_criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    def lookup_char_embedding(self, index, input, dim=1):
        '''
        index: [batch, 1/2]
        input: [batch, num_char, char_emb_size]
        '''
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
    
    def forward(self, sents, sent_lens, chars, actions, attributes, attr_lens, titles, title_len, masks, char_weights):
        '''
        sents: [batch, num_sent, sent_len]
        sent_lens: [batch, num_sent]
        chars, :[batch, num_sent]
        action: [batch, num_sent]
        attributes: [batch, num_sent, num_attr]
        title: [batch, num_sent, title_len + 2] # cat with action
        title_len: [batch]
        word_inputs: [batch, num_sent, sent_len]
        outputs: [batch, num_sent, sent_len]
        '''

        batch_size, num_sent, _ = sents.size()
        
        sent_embs = self.embedding(sents) #[batch, num_sent, sent_len, word_emb_size]
        
        attributes = [self.embedding(attribute) for attribute in attributes]
        R = self.char_emb_initializer(None, attributes, attr_lens) #[batch, num_char, word_emb_size]

#         attributes = torch.mean(self.embedding(attributes), dim=2) #[batch, num_sent, word_emb_size] #TODO
        
#         soc = self.embedding(torch.LongTensor(batch_size).fill_(6).unsqueeze(1).cuda())
#         attributes = torch.cat((soc, attributes), dim=1) #[batch, num_sent + 1, word_emb_size]
        
        char_idx_init = torch.LongTensor(batch_size).fill_(2).unsqueeze(1).cuda() #<soc> in char_list
        char_idx = torch.cat((char_idx_init, chars), dim=1) #[batch, num_sent + 1]
        
#         eot = torch.LongTensor(batch_size).fill_(7).unsqueeze(1).cuda()  [batch, 1]
#         title = torch.cat((title, eot), dim=1) #[batch, title_len + 1]
        titles = self.embedding(titles) #[batch, num_sent, title_len + 2, word_emb_size]
    
        
        action_emb = self.embedding(actions) #[batch, num_sent, word_emb_size]
        
        
        soa = self.embedding(torch.LongTensor(batch_size).fill_(7).unsqueeze(1).cuda())
        action_emb = torch.cat((soa, action_emb), dim=1) #[batch, num_sent + 1, word_emb_size]
#         word_inputs = self.embedding(word_inputs) #[batch, num_sent, sent_len, word_emb_size]
        
        h = self.t2h(self.dropout(torch.sum(titles[:, 0, :-2, :], dim=1)))
        
#         (hidden, cell) = self.init_hidden(batch_size)
        hidden = nn.init.xavier_uniform_(torch.FloatTensor(1, batch_size, self.word_hidden_size)).cuda() #self.init_hidden(batch_size)
    
    
        sent_mask = []
        for i in range(num_sent):
            sent_mask.append(torch.LongTensor([1 for _ in range(i + 1)]).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1).cuda())
        hs = []
        
        char_loss = action_loss = word_loss = loss = ppl = 0
        rec_char_logits = []
        for i in range(num_sent):
            
            char_emb = self.lookup_char_embedding(char_idx[:, i:i + 2], R, 1) #[batch, 2, char_emb_size]
            char_emb_i, char_emb_i_1 = char_emb[:, 0, :], char_emb[:, 1, :] #[batch, char_emb_size]
            
#             if i != 0:
#                 prev_sents = pack_padded_sequence(sent_embs[:, i - 1, :, :], sent_lens[:, i - 1], batch_first=True, enforce_sorted=False)
# #                 self.lstm.flatten_parameters()
# #                 _, (hidden, cell) = self.lstm(prev_sents, (hidden, cell)) #[2, batch, hidden_size]
#                 _, hidden = self.lstm(prev_sents, hidden)
    #         hidden = hidden.view(batch_size, -1) #[batch, 2 * hidden_size]
    #         cell = cell.view(batch_size, -1)
            S = hidden.view(batch_size, -1) #[batch, hidden_size]
            
            h = self.gru(torch.sum(titles[:, i, :-2, :], dim=1), char_emb_i, action_emb[:, i, :], R, S, h)
            
            
            hs.append(h)
            attn_h, _ = self.attn(h.unsqueeze(1), torch.stack(hs, 1), sent_mask[i]) #[batch, 1, sent_hidden_size]
            attn_h = attn_h.squeeze(1)
            
            
            
            selected_chars_emb, alignments, char_logits = self.char_selector(attn_h, R, masks) #[batch, char_emb_size], [batch, num_char], [batch, num_char]
            
            action_logits, action_id = self.action_selector(attn_h, char_emb_i_1) #[batch, vocab_size], [batch]
            
            _, logits, hidden, max_dec_len = self.sent_generator(sent_embs[:, i, :-1, :], (sent_lens[:, i] - 1).clamp_(1, self.max_sent_len), attn_h, char_emb_i_1, action_emb[:, i + 1, :], hidden.squeeze(0))
            
            h1 = torch.cat((attn_h, hidden.squeeze(0)), dim=-1) #[batch, sent_hidden_size + word_hidden_size]
            update = self.updater(alignments, h1) #[batch, num_char, char_emb_size]
            R = (1 - alignments).unsqueeze(2).mul(R) + update
#             R = R + self.beta * update
            

    #         cat = torch.cat((titles, action_emb), dim=1) #[batch, title_len + 2, word_emb_size]
#             cat = pack_padded_sequence(titles[:, i, :, :], title_len, batch_first=True, enforce_sorted=False)
#             logits, _ = self.sent_generator(cat, title_len, sent_embs[:, i, :-1, :], attributes[:, i, :], hidden, cell) #[batch * sent_len, vocab_size]

            action_loss += self.action_criterion(action_logits, actions[:, i])
            rec_char_logits.append(char_logits)
#             char_loss += self.char_criterion(char_logits, chars[:, i])
            local_word_loss = self.word_criterion(logits, sents[:, i, 1:max_dec_len + 1].contiguous().view(-1))
            word_loss += local_word_loss
            ppl += torch.exp(local_word_loss)
        
        rec_char_logits = torch.stack(rec_char_logits, dim=1) #[batch, num_sent, num_char]
        for i in range(batch_size):
            char_loss += F.cross_entropy(rec_char_logits[i, :, :], chars[i, :], weight=char_weights[i, :], ignore_index=0)
        
        char_loss /= batch_size
        action_loss /= num_sent
        word_loss /= num_sent
        loss = self.alpha2 * action_loss + self.alpha1 * char_loss + self.alpha3 * word_loss
        
        ppl /= num_sent
        return loss, char_loss, action_loss, word_loss, ppl
    
    
    def init_hidden(self, batch_size):
        hidden = nn.init.xavier_uniform_(next(self.lstm.parameters()).data.new(1, batch_size, self.hidden_size)).cuda()
        return hidden
#         hidden = nn.init.xavier_uniform_(next(self.lstm.parameters()).data.new(2, batch_size, self.hidden_size)).cuda()
#         cell =  nn.init.xavier_uniform_(next(self.lstm.parameters()).data.new(2, batch_size, self.hidden_size)).cuda()
#         return (hidden, cell)
    
    def find_first_eos(self, s):
        # sents [len]
        
        length = len(s)
        idx = 0
        for w in s:
            if w.item() == 3:
                break
            idx += 1
        if idx != length:
            idx += 1
        return s[:idx]
    
    def generate(self, sents, sent_lens, chars, actions, attributes, attr_lens, titles, title_len, masks, char_weights):
        '''
        sents: [batch, num_sent, sent_len]
        sent_lens: [batch, num_sent]
        action: [batch, num_sent]
        attributes: [batch, num_sent, num_attr]
        title: [batch, num_sent, title_len + 2] # cat with action
        title_len: [batch]
        word_inputs: [batch, num_sent, sent_len]
        outputs: [batch, num_sent, sent_len]
        '''
        batch_size, num_sent, _ = sents.size()
        
        
        attributes = [self.embedding(attribute) for attribute in attributes]
        R = self.char_emb_initializer(None, attributes, attr_lens) #[batch, num_char, word_emb_size]
        
        char_idx_init = torch.LongTensor(batch_size).fill_(2).unsqueeze(1).cuda() #<soc> in char_list
        char_idx = torch.cat((char_idx_init, chars), dim=1) #[batch, num_sent + 1]
#         char_idx = torch.LongTensor(batch_size).fill_(2).unsqueeze(1).cuda()
        
        action_idx = torch.LongTensor(batch_size).fill_(7).cuda()

        titles = self.embedding(titles) #[batch, num_sent, title_len + 2, word_emb_size]
#         action_emb = self.embedding(actions).unsqueeze(1) #[batch, 1, word_emb_size]
        word_inputs = self.embedding(torch.LongTensor(batch_size).fill_(2).unsqueeze(1).cuda()) #[batch, 1, word_emb_size]
    
        h = self.t2h(self.dropout(torch.sum(titles[:, 0, :-2, :], dim=1)))
        hidden = nn.init.xavier_uniform_(torch.FloatTensor(1, batch_size, self.word_hidden_size)).cuda()
        
        sent_mask = []
        for i in range(num_sent):
            sent_mask.append(torch.LongTensor([1 for _ in range(i + 1)]).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1).cuda())
        hs = []
        
        
        selected_chars = []
        action_ids = []
        all_outputs = []
        char_loss = action_loss = word_loss = loss = ppl = 0
        rec_char_idx = -1
        rec_action_idx = set()
        rec_char_logits = []
        for i in range(num_sent):
            char_emb = self.lookup_char_embedding(char_idx[:, i:i + 2], R, 1) #[batch, 2, char_emb_size]
            char_emb_i, char_emb_i_1 = char_emb[:, 0, :], char_emb[:, 1, :] #[batch, char_emb_size]
#             char_emb_i = self.lookup_char_embedding(char_idx, R, 1).squeeze(1) #[batch, char_emb_size]
            action_emb_i = self.embedding(action_idx)
            
            
            S = hidden.view(batch_size, -1) #[batch, 2 * hidden_size]
            
            h = self.gru(torch.sum(titles[:, i, :-2, :], dim=1), char_emb_i, action_emb_i, R, S, h)
            
            hs.append(h)
            attn_h, _ = self.attn(h.unsqueeze(1), torch.stack(hs, 1), sent_mask[i]) #[batch, 1, sent_hidden_size]
            attn_h = attn_h.squeeze(1)
            
            selected_chars_emb, alignments, char_logits = self.char_selector(attn_h, R, masks)
            
#             char_idx = torch.argmax(alignments, -1).unsqueeze(1) #[batch, 1]
#             char_embedding = self.lookup_char_embedding(char_idx, R, 1).squeeze(1) #[batch, char_emb_size]
            
            action_logits, action_idx = self.action_selector(attn_h, char_emb_i_1) #[batch, verb_vocab_size], [batch]
        
            
            _, indexes = torch.topk(F.log_softmax(action_logits, dim=-1).squeeze(0), 20)
            for idx in indexes:
                if idx.item() not in rec_action_idx:
                    action_idx = idx.unsqueeze(0)
                    break
#             rec_char_idx = char_idx[0, i + 1].item()
#             rec_char_idx = char_idx.item()
            rec_action_idx.add(action_idx.item())
            
            
            action_embedding = self.embedding(action_idx) #[batch, word_emb_size]

            outputs, logits, hidden = self.sent_generator.generate_beam_search(word_inputs, attn_h, char_emb_i_1, action_embedding, self.embedding, hidden.squeeze(0)) #[batch, gen_sent_len], #[batch, gen_sent_len, vocab_size]

            h1 = torch.cat((attn_h, hidden.squeeze(0)), dim=-1) #[batch, sent_hidden_size + word_hidden_size]
            update = self.updater(alignments, h1) #[batch, num_char, char_emb_size]
            R = (1 - alignments).unsqueeze(2).mul(R) + update
#             R = R + self.beta * update
            
            rec_char_logits.append(char_logits)
#             char_loss += self.char_criterion(char_logits, chars[:, i])
            action_loss += self.action_criterion(action_logits, actions[:, i])
            sent_output = sents[:, i, 1:].contiguous().view(-1)
#             local_word_loss = self.word_criterion(logits.view(-1, self.vocab_size)[:sent_output.size(0), :], sent_output)
#             word_loss += local_word_loss
#             ppl += torch.exp(local_word_loss)
            
            
            selected_char = torch.argmax(alignments.squeeze(0)).item()
            action_id = action_idx.item()
            outputs = list(self.find_first_eos(outputs.squeeze(0)))
             
            selected_chars.append(selected_char)
            action_ids.append(action_id)
            all_outputs.append(outputs)
        
        
        rec_char_logits = torch.stack(rec_char_logits, dim=1) #[batch, num_sent, num_char]
        for i in range(batch_size):
            char_loss += F.cross_entropy(rec_char_logits[i, :, :], chars[i, :], weight=char_weights[i, :], ignore_index=0)
        
        char_loss /= batch_size
        action_loss /= num_sent
        word_loss /= num_sent
        loss = self.alpha1 * char_loss + self.alpha2 * action_loss + self.alpha3 * word_loss
        ppl /= num_sent
        
        return selected_chars, action_ids, all_outputs, loss, char_loss, action_loss, word_loss, ppl