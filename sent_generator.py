import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from queue import PriorityQueue
import operator

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

class BeamSearchNode(object):
    def __init__(self, hiddenstate, hs, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.prevHidden = hs
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    def __lt__(self, other):
        return self.logp < other.logp


class Sent_Generator(nn.Module):
    
    def __init__(self, hps):
        super(Sent_Generator, self).__init__()
        
        char_emb_size = hps.char_emb_size
        word_emb_size = hps.word_emb_size
        verb_emb_size = hps.verb_emb_size
        self.word_hidden_size = hps.word_hidden_size
        sent_hidden_size = hps.sent_hidden_size
        vocab_size = hps.vocab_size
        dropout = hps.dropout
        self.gen_sent_len = hps.gen_sent_len
        
        self.compat_h = nn.Linear(sent_hidden_size + char_emb_size + verb_emb_size + self.word_hidden_size, self.word_hidden_size)
#         self.compat_c = nn.Linear(sent_hidden_size + char_emb_size + verb_emb_size, self.word_hidden_size)
        self.gru = nn.GRU(word_emb_size, self.word_hidden_size, 1, batch_first=True)
#         self.lstm = nn.LSTM(word_emb_size, self.word_hidden_size, 1, batch_first=True)
        self.attn = GlobalAttention(self.word_hidden_size, self.word_hidden_size, dropout)
        self.fc = nn.Linear(self.word_hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

    
    def forward(self, inputs, sent_lens, hidden, chars, action_embedding, sent_h):
        '''
        inputs: [batch, sent_len, word_emb_size]
        sent_lens: [batch]
        hidden: [batch, sent_hidden_size]
        chars: [batch, char_emb_size]
        action_embedding: [batch, verb_emb_size]
        sent_h: [batch, word_hidden_size]
        '''
        batch_size, sent_len, _ = inputs.size()
#         hidden = hidden.squeeze(0) #[batch, sent_hidden_size]
        cat = torch.cat((hidden, chars, action_embedding, sent_h), 1) #[batch, sent_hidden_size + char_emb_size + verb_emb_size]
        h = self.compat_h(self.dropout1(cat)).unsqueeze(0) #[1, batch, word_hidden_size]
#         c = self.compat_c(self.dropout2(cat)).unsqueeze(0)
        
        inputs = pack_padded_sequence(inputs, sent_lens, batch_first=True, enforce_sorted=False)
        out, h = self.gru(inputs, h) #[batch, sent_len, word_hidden_size], [1, batch, word_hidden_size]
#         c = nn.init.xavier_uniform_(torch.FloatTensor(1, inputs.size(0), self.word_hidden_size)).cuda()
#         out, (h, c) = self.lstm(inputs, (h, c)) #[batch, sent_len, word_hidden_size]
        
        out, dec_lens = pad_packed_sequence(out, batch_first=True)
        max_dec_len = max(dec_lens)

        mask = []
        for i in range(max_dec_len):
            mask.append([1 for _ in range(i + 1)] + [0 for _ in range(max_dec_len - i - 1)])
        mask = torch.LongTensor(mask).unsqueeze(0).expand(batch_size, -1, -1).cuda()
        out, _ = self.attn(out, out, mask)

        out = self.dropout(out.contiguous().view(-1, self.word_hidden_size)) #[batch * sent_len, word_hidden_size]
        out = self.fc(out) #[batch * sent_len, vocab_size]
        softmax = F.softmax(out, dim=-1)
        
        return softmax, out, h, max_dec_len #[batch * sent_len, vocab_size], [batch * sent_len, vocab_size], [1, batch, word_hidden_size]
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
#         hidden = weight.new(1, batch_size, self.word_hidden_size).zero_().cuda()
        hidden = nn.init.xavier_uniform_(weight.new(1, batch_size, self.word_hidden_size)).cuda()
        return hidden
    
    def generate(self, inputs, hidden, chars, action_embedding, embedding, sent_h):
        '''
        inputs: [batch, 1, word_emb_size]
        hidden: [batch, sent_hidden_size]
        chars: [batch, char_emb_size]
        action_embedding: [batch, verb_emb_size]
        '''
        batch_size = chars.size(0)
#         hidden = hidden.squeeze(0) #[batch, sent_hidden_size]
        cat = torch.cat((hidden, chars, action_embedding, sent_h), 1) #[batch, sent_hidden_size + char_emb_size + verb_emb_size]
        h = self.compat_h(cat) #[batch, word_hidden_size]
        h0 = h.unsqueeze(0) #[1, batch, word_hidden_size]
        h = h0
        
        
        mask = []
        for i in range(self.gen_sent_len):
            mask.append(torch.LongTensor([1 for _ in range(i + 1)]).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1).cuda())
        hs = []
        
        
        outputs = []
        outs = []
        for i in range(self.gen_sent_len - 1):
            out, h = self.gru(inputs, h) #[batch, 1, word_hidden_size], [1, batch, word_hidden_size]
            
            hs.append(out)
            out, _ = self.attn(out, torch.cat(hs, 1), mask[i]) #[batch, 1, word_hidden_size]

            
            out = out.contiguous().view(-1, self.word_hidden_size)
            out = self.fc(out) #[batch, vocab_size]
            outs.append(out)
            softmax = F.softmax(out, dim=-1) #[batch, vocab_size]
            inputs = torch.argmax(softmax, dim=-1, keepdim=True) #[batch, 1]
            outputs.append(inputs)
            inputs = embedding(inputs) #[batch, 1, word_emb_size]
        
        outputs = torch.cat(outputs, 1) #[batch, gen_sent_len]
        outs = torch.stack(outs, 1) #[batch, gen_sent_len, vocab_size]
        
        sos = torch.LongTensor(batch_size).fill_(2).unsqueeze(1).cuda() #[batch, 1]
        _, h = self.gru(embedding(torch.cat((sos, self.find_first_eos(outputs.squeeze(0)).unsqueeze(0)), dim=1)), h0)
        return outputs, outs, h
    
    def generate_beam_search(self, inputs, hidden, chars, action_embedding, embedding, sent_h):
        '''
        inputs: [batch, 1, word_emb_size]
        hidden: [batch, sent_hidden_size]
        chars: [batch, char_emb_size]
        action_embedding: [batch, verb_emb_size]
        '''
        batch_size = chars.size(0)
        beam_width = 5
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []
        final_hiddens = []
        
        cat = torch.cat((hidden, chars, action_embedding, sent_h), 1) #[batch, sent_hidden_size + char_emb_size + verb_emb_size]
        decoder_hiddens = self.compat_h(cat) #[batch, word_hidden_size]
        
        mask = []
        for i in range(100):
            mask.append(torch.LongTensor([1 for _ in range(i + 1)]).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1).cuda())
        
        for idx in range(batch_size):
            
            decoder_hidden = decoder_hiddens[idx, :].unsqueeze(0).unsqueeze(0)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[2]]).cuda()

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, [], None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h
                hs = n.prevHidden
                leng = n.leng

                if (n.wordid.item() == 3 or n.leng == self.gen_sent_len) and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                
                decoder_output, decoder_hidden = self.gru(embedding(decoder_input), decoder_hidden)
                new_hs = hs + [decoder_output]
                
                out, _ = self.attn(decoder_output, torch.cat(new_hs, 1), mask[leng - 1]) #[batch, 1, word_hidden_size]
                out = out.contiguous().view(-1, self.word_hidden_size)
                out = self.fc(out) #[batch, vocab_size]
                decoder_output = F.log_softmax(out, dim=-1) #[batch, vocab_size]

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, new_hs, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            final_hidden = None
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                final_hidden = n.h
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(torch.LongTensor(utterances[0]).cuda())
            final_hiddens.append(final_hidden)

        return torch.stack(decoded_batch, dim=0), None, torch.cat(final_hiddens, dim=1) #[batch, sent_len], [1, batch, word_hidden_size]
        
        
    def find_first_eos(self, s):
        # sents [len]
        
        length = len(s)
        idx = 0
        for w in s:
            if w.item() == 3:
                break
            idx += 1
        return s[:idx]