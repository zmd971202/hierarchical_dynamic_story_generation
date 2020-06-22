import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from collections import Counter
import random
import numpy as np
import re
import unidecode

class WithReplacementRandomSampler(Sampler):
    """Samples elements randomly, with replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # generate samples of `len(data_source)` that are of value from `0` to `len(data_source)-1`
        samples = torch.LongTensor(len(self.data_source))
        samples.random_(0, len(self.data_source))
        return iter(samples)

    def __len__(self):
        return len(self.data_source)

class Dataset(data.Dataset):
    
    def __init__(self, titles, sents, all_chars, attributes, chars, actions, word2idx, hps):
        '''
        titles: str [num_summary, title_len] 
        sents: str [num_summary, num_sent, sent_len]
        all_chars: str [num_summary, num_char, char_len]
        chars: int [num_summary, num_sent]
        actions: str [num_summary, num_sent]
        sorted
        '''
        self.titles = titles
        self.sents = sents
        self.all_chars = all_chars #[]
        self.attributes = attributes #[]
        self.chars = chars
        self.actions = actions
    
        self.word2idx = word2idx
        self.max_sent_len = hps.max_sent_len
        self.max_num_sent = hps.max_num_sent
        
#         for char, attribute in zip(chars, attributes):
#             attr = []
#             for c in char:
#                 attr.append(attribute[c])
#             self.attributes.append(attr)
        

    def __len__(self):

        return len(self.titles)

    def __getitem__(self, index):
        
        title = self.titles[index] + ['<eot>']
        sent = self.sents[index]
        all_char = self.all_chars[index]
        attrs = self.attributes[index]
        char = self.chars[index]
        actions = self.actions[index]
        
        title = [self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in title]
        sent = self.sent2id(sent)
        all_char = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in c] for c in all_char]
        attrs = [[self.word2idx[w] if w in self.word2idx else self.word2idx['<unk>'] for w in attr] for attr in attrs]
        char = char[:self.max_num_sent]
        actions = [self.word2idx[action] if action in self.word2idx else self.word2idx['<unk>'] for action in actions]
        
        return title, sent, attrs, actions, char, all_char
    
    def sent2id(self, sent):
        # sent
        seqs = []
        for s in sent[:self.max_num_sent]:
            seq = []
            seq.append(self.word2idx['<sos>'])
            for w in s:
                if w in self.word2idx:
                    seq.append(self.word2idx[w])
                else:
                    seq.append(self.word2idx['<unk>'])
            seq.append(self.word2idx['<eos>']) 
            seq = seq[:self.max_sent_len]
            seqs.append(seq)
        return seqs

def add_vocab(titles, sents):
    words = []
    for t in titles:
        words += t
    for sent in sents:
        for s in sent[:40]:
            words += [w for w in s if '<char' not in w]
    return words
    
    
def make_vocab(hps):
    words = []
    titles, sents, _, _, _, actions = load_data(hps.train_data_path, hps)
    words += add_vocab(titles, sents)


    titles, sents, all_chars, _, _, actions = load_data(hps.valid_data_path, hps)
    words += add_vocab(titles, sents)


    c = Counter(words)
    top_k_words = sorted(c.keys(), reverse=True, key=c.get)
    words = ['<pad>', '<unk>', '<sos>', '<eos>', '<noc>', '<eot>', '<soc>', '<soa>', '<attr>', '<eoc>'] + ['<char{}>'.format(i) for i in range(hps.max_num_char - 3)] +\
            [w for w in top_k_words if c[w] > 5] #[:hps.vocab_size]
    print('make vocab size {}'.format(len(words)))
    word2idx = {w:i for i, w in enumerate(words)}
    idx2word = {i:w for i, w in enumerate(words)}
    
    open(hps.test_path + hps.vocab_path, 'w', encoding='utf-8').write('\n'.join([w + '\t' + str(word2idx[w]) for w in word2idx]))
    return word2idx, idx2word

def load_vocab(hps):
    lines = open(hps.test_path + hps.vocab_path, 'r').read().split('\n')
    
    word2idx, idx2word = {}, {}
    for l in lines:
        w, idx = l.split('\t')
        idx = int(idx)
        word2idx[w] = idx
        idx2word[idx] = w
    print('load vocab size {}'.format(len(word2idx)))
    
    return word2idx, idx2word

def collate_fn(data):
    def padding(arr, length):
        return arr[:length] + [0 for _ in range(length - len(arr))]
    def padding_one(arr, length):
        return arr[:length] + [1 for _ in range(length - len(arr))]
    def padding_arr(arr, length):
        return arr[:length] + [[0 for _ in range(len(arr[0]))] for _ in range(length - len(arr))]

#     data = sorted(data, lambda x : len(x[1]), reverse=True)
    titles, sents, attributes, actions, chars, all_chars = zip(*data)  # tuples
    max_title_len = max(len(t) for t in titles)
    max_num_sent = max(len(sent) for sent in sents)
    max_num_char = 13
    max_num_attr = 10
    max_sent_len = max(max(len(s) for s in sent) for sent in sents)
    max_char_len = max(max(len(char) for char in all_char) for all_char in all_chars)
    
    title_len = torch.LongTensor([len(t) + 1 for t in titles])
    
    titles = torch.LongTensor([padding_arr([padding(t + [a], max_title_len + 1) for a in action], max_num_sent) for t, action in zip(titles, actions)])
#     sent_inputs = torch.LongTensor([[t for _ in range(max_num_sent)] for t in titles])  #[batch, max_num_sent, max_title_len]
    
    sent_lens = torch.LongTensor([padding_one([len(s) for s in sent], max_num_sent) for sent in sents]) #[batch, maxnum_sent]
    sents = [padding_arr([padding(s, max_sent_len) for s in sent], max_num_sent) for sent in sents] #[batch, max_num_sent, max_sent_len]
    #word_inputs, outputs = torch.LongTensor([[s[:-1] for s in sent] for sent in sents]), torch.LongTensor([[s[1:] for s in sent] for sent in sents])
    sents = torch.LongTensor(sents)
    
    char_weights = []
    for char in chars:
        cnts = [0] * max_num_char
        for c in char:
            cnts[c] += 1
        SUM = sum(cnts)
        w = [SUM / cnt if cnt != 0 else 2 * SUM for cnt in cnts]
        w[1] = 1
        char_weights.append(w)
    char_weights = torch.FloatTensor(char_weights)  
    chars = torch.LongTensor([padding(char, max_num_sent) for char in chars])
    masks = torch.FloatTensor([[int(any(char)) for char in all_char] for all_char in all_chars])#.bool()
    all_chars = [padding_arr([padding(char, max_char_len) for char in all_char], max_num_char) for all_char in all_chars]
    all_chars = torch.LongTensor(all_chars)
    
    
    attr_lens = [] #[13, batch]
    new_attributes = [] #[13, batch]
    for i in range(max_num_char):
        attr_len = [min(len(attribute[i]), max_num_attr) for attribute in attributes]
        max_attr_len = max(attr_len)
        attr = []
        for attribute in attributes:
            attr.append(padding(attribute[i], max_attr_len))
        new_attributes.append(torch.LongTensor(attr))
        attr_lens.append(torch.LongTensor(attr_len))
        
#     attributes = torch.LongTensor([padding_arr([padding(attr, max_num_attr) for attr in attribute], max_num_sent) for attribute in attributes])
    
    actions = torch.LongTensor([padding(action, max_num_sent) for action in actions]) #[batch, max_num_sent]
    
    return sents, sent_lens, chars, actions, all_chars, new_attributes, attr_lens, titles, title_len, masks, char_weights#, word_inputs, outputs

# def segment_data(titles, sents, all_chars, attributes, chars, actions):
#     new_titles, new_sents, new_all_chars, new_attributes, new_chars, new_actions = [], [], [], [], [], []
#     for t, sent, all_char, attr, char, action in zip(titles, sents, all_chars, attributes, chars, actions):
        
    

def load_data(data_path, hps):
    lines = [l for l in open(data_path, 'r').read().split('\n') if len(l) > 0]
#     lines = [lines[4]]
#     random.seed(1234)
    lines = random.sample(lines, 10)
#     lines = sorted(lines, key=lambda x:len(x.split('\t')[4].split(' ||| ')), reverse=True)
    titles = []
    all_chars = []
    chars = []
    attributes = []  #TODO
    actions = []
    sents = []
    for l in lines:
        l = unidecode.unidecode(l)
        l = re.sub('[0-9]+', '<num>', l)
        id, title, all_char, char, sent = l.split('\t')
        titles.append(title.lower().split())
        sent = sent.lower()
        attr = [[w.lower() for w in c.split('@@@')[1].split(' ') if w != ''] for c in all_char.split(' ; ')]
#         attributes.append(attr)
        all_char = [c.split('@@@')[0].lower() for c in all_char.split(' ; ')]
        new_all_char, new_attr = [''] * (hps.max_num_char - 3), [['<char{}>'.format(i), '<attr>'] for i in range(hps.max_num_char - 3)]
        idxs = random.sample([i for i in range(hps.max_num_char - 3)], len(all_char))
        for idx, c, a in zip(idxs, all_char, attr):
            new_all_char[idx] = c
            new_attr[idx] += a
            sent = sent.replace(c, '<char{}>'.format(idx))
        attributes.append([['<pad>'], ['<noc>'], ['<soc>']] + new_attr)
        all_char = ['<pad>', '<noc>', '<soc>'] + new_all_char
#         all_char = all_char.split(' ; ')
#         pos = int(random.random() * (len(all_char) + 1))
#         all_char = ['<pad>'] + all_char[:pos] + ['NOCHAR'] + all_char[pos:]
#         try:
        chars.append([all_char.index(s.split(' ; ')[0].lower()) if s.split(' ; ')[0] != '<noc>' else 1 for s in char.split(' ||| ')])
#         except Exception as e:
#             print(e)
#             print(all_char)
#             print(char)
        all_chars.append([c.split() for c in all_char])
        actions.append([s.split(' ; ')[1].lower() for s in char.split(' ||| ')])
        sents.append([s.split() for s in sent.split(' ||| ')])
    return titles, sents, all_chars, attributes, chars, actions

def make_weights_for_balanced_classes(chars):
    counts = [char.count(1) / len(char) for char in chars]
    SUM = sum(counts)
    weights = [SUM / count if count != 0 else 2 * SUM for count in counts]
    return weights
    
    
def load_train_data(word2idx, hps):
    # train data already splitted
    char_cnts, action_cnts = [0] * hps.max_num_char, [0] * len(word2idx)
    titles, sents, all_chars, attributes, chars, actions = load_data(hps.train_data_path, hps)
    for char in chars:
        for c in char[:hps.max_num_sent]:
            char_cnts[c] += 1
    for action in actions:
        for a in action[:hps.max_num_sent]:
            idx = word2idx[a] if a in word2idx else word2idx['<unk>']
            if idx != 0:
                action_cnts[idx] += 1
    print('train data length: {}'.format(len(titles)))
    train_data = Dataset(titles, sents, all_chars, attributes, chars, actions, word2idx, hps)
    
#     weights = make_weights_for_balanced_classes(chars)
#     sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_data, batch_size=hps.batch_size, 
                              shuffle=False, collate_fn=collate_fn, drop_last=True, sampler=WithReplacementRandomSampler(train_data))
    
    # valid data
    titles, sents, all_chars, attributes, chars, actions = load_data(hps.valid_data_path, hps)
    for char in chars:
        for c in char[:hps.max_num_sent]:
            char_cnts[c] += 1
    for action in actions:
        for a in action[:hps.max_num_sent]:
            idx = word2idx[a] if a in word2idx else word2idx['<unk>']
            if idx != 0:
                action_cnts[idx] += 1
    print('valid data length: {}'.format(len(titles)))
    valid_data = Dataset(titles, sents, all_chars, attributes, chars, actions, word2idx, hps)
    
#     weights = make_weights_for_balanced_classes(chars)
#     sampler = WeightedRandomSampler(weights, len(weights))
    
    valid_loader = DataLoader(valid_data, batch_size=hps.batch_size, 
                              shuffle=True, collate_fn=collate_fn, drop_last=True)#, sampler=sampler)
    
    SUM = sum(char_cnts)
    char_weights = [SUM / cnt if cnt != 0 else 2 * SUM for cnt in char_cnts]
    SUM = sum(action_cnts)
    action_weights = [SUM / cnt if cnt != 0 else 2 * SUM for cnt in action_cnts]
    return train_loader, valid_loader, char_weights, action_weights

    
def load_test_data(word2idx, hps):
    # test data
    titles, sents, all_chars, attributes, chars, actions = load_data(hps.test_data_path, hps)
    print('test data length: {}'.format(len(titles)))
    test_data = Dataset(titles, sents, all_chars, attributes, chars, actions, word2idx, hps)
    test_loader = DataLoader(test_data, batch_size=1, 
                              shuffle=False, collate_fn=collate_fn, drop_last=False)
    anony2names = [{'<char{}>'.format(i):' '.join(c) for i, c in enumerate(all_char[3:])} for all_char in all_chars]
    return test_loader, anony2names