import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
import bleu
import os
import matplotlib.pyplot as plt

from model import Model
# from test_model import Model
from data import make_vocab, load_train_data, load_test_data, load_vocab

from collections import namedtuple
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

hps_dict = {
    
    'cuda_visible_devices': '0',
    'test_path':'dumped/test6/',
    'epochs':200,
    'test':False,
    'checkpoint_every':1,
    'print_every':200,
    'lr': 0.0001,
    'factor': 0.5,
    'patience': 1,
    'weight_decay':1e-5,
    'l1_lambda':0, #1e-5,
    'clip': 5.0,
    
    # data
    'max_sent_len':25, #need to be equal to gen_sent_len
    'max_num_sent':10, #20, #40
    'max_num_char':13,
    'max_num_attr':8, #TODO
    'vocab_path':'vocab.txt',
    'train_data_path':'data/train_data.txt',
    'valid_data_path':'data/valid_data.txt',
    'test_data_path':'data/test_data.txt',
    'gen_output_path':'gen_output.txt',
    'ref_output_path':'ref_output.txt',
    
    # model
    'batch_size':8,
    'vocab_size':66979, # need to be set accordingly
    'dropout':0.1,
    'alpha1':0.1,
    'alpha2':0.1,
    'alpha3':0.8,
    'beta':0.5,
    'save_path':'models/best.model',
    
    # sentence
    'sent_hidden_size':512,
    'gen_sent_len':25, #25,
    
    # char
    'char_emb_size': 512,
    'char_hidden_size':512,
    
    # word
    'word_hidden_size':512,
    'word_emb_size':512,        # word_emb_size = char_emb_size
    'verb_emb_size':512

           }
hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
hps = hps._replace(cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])


parser = argparse.ArgumentParser()
parser.add_argument("--pre_train", action='store_true', help="pre_train")
parser.add_argument("--train", action='store_true', help="train")
parser.add_argument("--test", action='store_true', help="test")
parser.add_argument("--reload", action='store_true', help="reload")
parser.add_argument("--reload_epoch", type=int, help="reload_epoch", default=-1)
args = parser.parse_args()


def cal_reg_loss(model):
    reg_loss = 0
    for param in model.parameters():
        reg_loss += torch.sum(torch.abs(param))
    return reg_loss
    
def evaluate(gen_chars, gen_actions, gen_outputs, ref_chars, ref_actions, ref_outputs, idx2word):
    '''
    bleu
    ppl
    char_acc
    '''
    # char
    total_sum = total_length = 0
    for gc, rc in zip(gen_chars, ref_chars):
        eq = np.equal(gc, rc)
        total_sum += sum(eq)
        total_length += len(eq)
    print('char_acc:{}'.format((total_sum / total_length) * 100), flush=True)
    
    # action
    total_sum = total_length = 0
    for ga, ra in zip(gen_actions, ref_actions):
        eq = np.equal(ga, ra)
        total_sum += sum(eq)
        total_length += len(eq)
    print('action_acc:{}'.format((total_sum / total_length) * 100), flush=True)
    
    gen_outputs = [' '.join([' '.join([idx2word[id.item()] for id in s]) for s in outputs]) for outputs in gen_outputs]
    ref_outputs = [[' '.join([' '.join([idx2word[id.item()] for id in s]) for s in outputs])] for outputs in ref_outputs]   # TODO restore name?
    scores, _ = bleu.corpus_bleu(gen_outputs, ref_outputs, max_n=4)
    final_bleu, bleu1, bleu2, bleu3, bleu4 = scores
    print('bleu1:{}, bleu2:{}, bleu3:{}, bleu4:{}'.format(bleu1 * 100, bleu2 * 100, bleu3 * 100, bleu4 * 100), flush=True)

def find_first_eos(s):
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

def restore_name(text, anony2name):
    for a in anony2name:
        name = anony2name[a]
        text = text.replace(a, name)
    return text

def print_output(titles, gen_chars, gen_actions, gen_outputs, ref_chars, ref_actions, ref_outputs, total_all_chars, idx2word, anony2names, hps):
    titles = [' '.join([idx2word[t] for t in title]) for title in titles]
    
    new_gen_chars, new_ref_chars = [], []
    for gc, rc, all_chars in zip(gen_chars, ref_chars, total_all_chars):
        new_gen_chars.append([' '.join([idx2word[i.item()] for i in all_chars[c]]) for c in gc])
        new_ref_chars.append([' '.join([idx2word[i.item()] for i in all_chars[c]]) for c in rc])
    
    new_gen_actions, new_ref_actions = [], []
    for ga, ra in zip(gen_actions, ref_actions):
        new_gen_actions.append([idx2word[a] for a in ga])
        new_ref_actions.append([idx2word[a.item()] for a in ra])
    
    new_gen_outputs = [[restore_name(' '.join([idx2word[id.item()] for id in s]), anony2name) for s in outputs] for outputs, anony2name in zip(gen_outputs, anony2names)]
    new_ref_outputs = [[restore_name(' '.join([idx2word[id.item()] for id in find_first_eos(s)]), anony2name) for s in outputs] for outputs, anony2name in zip(ref_outputs, anony2names)]
    
    gen_strings = []
    for t, gc, ga, go in zip(titles, new_gen_chars, new_gen_actions, new_gen_outputs):
        assert len(gc) == len(ga) and len(ga) == len(go)
        string = t + '\t' + ' ||| '.join([c + ' ; ' + a for c, a in zip(gc, ga)]) + '\t' + ' ||| '.join(go)
        gen_strings.append(string)
    open(hps.test_path + hps.gen_output_path, 'w', encoding='utf-8').write('\n'.join(gen_strings))
    
    ref_strings = []
    for t, rc, ra, ro in zip(titles, new_ref_chars, new_ref_actions, new_ref_outputs):
        assert len(rc) == len(ra) and len(ra) == len(ro)
        string = t + '\t' + ' ||| '.join([c + ' ; ' + a for c, a in zip(rc, ra)]) + '\t' + ' ||| '.join(ro)
        ref_strings.append(string)
    open(hps.test_path + hps.ref_output_path, 'w', encoding='utf-8').write('\n'.join(ref_strings))
    

def test(model, test_loader, idx2word, anony2names, hps):
    model.eval()
    gen_chars, gen_actions, gen_outputs = [], [], []
    ref_chars, ref_actions, ref_outputs = [], [], []
    rec_titles, total_all_chars = [], []
    total_ppl = num_data = 0
    for i, test_data in enumerate(test_loader):
#         if i == 100:
#             break
        sents, sent_lens, chars, actions, all_chars, attributes, attr_lens, titles, title_len, masks, char_weights = test_data

        total_all_chars.append([[c for c in char if c != 0] for char in list(all_chars.squeeze(0))])
        rec_titles.append(titles.squeeze(0)[0][:-2].tolist())
        
        sents, sent_lens, chars, actions= sents.cuda(), sent_lens.cuda(), chars.cuda(), actions.cuda()
        titles, title_len, masks, char_weights = titles.cuda(), title_len.cuda(), masks.cuda(), char_weights.cuda()
        attributes = [attribute.cuda() for attribute in attributes]
        attr_lens = [attr_len.cuda() for attr_len in attr_lens]
#         word_inputs, outputs = word_inputs.cuda(), outputs.cuda()
        with torch.no_grad():
            selected_chars, action_ids, all_outputs, loss, char_loss, action_loss, word_loss, ppl = model.generate(sents, sent_lens, chars, 
                                                                                                                   actions, attributes, attr_lens, 
                                                                                                                   titles, title_len, masks, char_weights) #[num_sent], [num_sent], [num_sent, sent_len]
        
        gen_chars.append(selected_chars)
        gen_actions.append(action_ids)
        gen_outputs.append(all_outputs)
        ref_chars.append(list(chars.squeeze(0)))
        ref_actions.append(list(actions.squeeze(0)))
        ref_outputs.append(list(sents[:, :, 1:].squeeze(0)))
#         total_ppl += ppl.item()
        num_data += 1
    evaluate(gen_chars, gen_actions, gen_outputs, ref_chars, ref_actions, ref_outputs, idx2word)
    print('ppl:{}'.format(total_ppl / num_data), flush=True)
    
    print('printing outputs')

    print_output(rec_titles, gen_chars, gen_actions, gen_outputs, ref_chars, ref_actions, ref_outputs, total_all_chars, idx2word, anony2names, hps)

def valid(model, valid_loader, idx2word, hps):
    model.eval()
    gen_chars, gen_actions, gen_outputs = [], [], []
    ref_chars, ref_actions, ref_outputs = [], [], []
    total_loss = total_char_loss = total_action_loss = total_word_loss = total_ppl = num_data = 0
    for i, valid_data in enumerate(valid_loader):
#         if random.random() > 0.1:
#             continue
#         sents, sent_lens, chars, actions, attributes, titles, title_len = valid_data
#         sents, sent_lens, chars, actions= sents.cuda(), sent_lens.cuda(), chars.cuda(), actions.cuda()
#         attributes, titles, title_len = attributes.cuda(), titles.cuda(), title_len.cuda()


        sents, sent_lens, chars, actions, _, attributes, attr_lens, titles, title_len, masks, char_weights = valid_data
        sents, sent_lens, chars, actions= sents.cuda(), sent_lens.cuda(), chars.cuda(), actions.cuda()
        titles, title_len, masks, char_weights = titles.cuda(), title_len.cuda(), masks.cuda(), char_weights.cuda()
        attributes = [attribute.cuda() for attribute in attributes]
        attr_lens = [attr_len.cuda() for attr_len in attr_lens]
        
        with torch.no_grad():
            loss, char_loss, action_loss, word_loss, ppl = model(sents, sent_lens, chars, actions, attributes, attr_lens, titles, title_len, masks, char_weights)
#             _, _, _, loss, char_loss, action_loss, word_loss, ppl = model.generate(sent_inputs, word_inputs, outputs, all_chars, attributes, masks, chars, actions)
#             selected_chars, action_ids, all_outputs, ppl = model.generate(sent_inputs, outputs, all_chars, masks) #[num_sent], [num_sent], [num_sent, sent_len]
#         gen_chars.append(selected_chars)
#         gen_actions.append(action_ids)
#         gen_outputs.append(all_outputs)
#         ref_chars.append(list(chars.squeeze(0)))
#         ref_actions.append(list(actions.squeeze(0)))
#         ref_outputs.append(list(outputs.squeeze(0)))
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_char_loss += char_loss.item()
        total_word_loss += word_loss.item()
        total_ppl += ppl.item()
        num_data += 1
#     evaluate(gen_chars, gen_actions, gen_outputs, ref_chars, ref_actions, ref_outputs, idx2word)
    print('valid avg_loss {} char_loss {} action_loss {} word_loss {} ppl {}'.format(total_loss / num_data, total_char_loss / num_data, total_action_loss / num_data, 
                                                                                     total_word_loss / num_data, total_ppl / num_data), flush=True)
#     print('valid loss:{}'.format(total_loss / num_data), flush=True)
    return total_loss / num_data
    

# train
def train(model, opt, train_loader, valid_loader, idx2word, hps):
    epochs, checkpoint_every, print_every, reload_epoch, l1_lambda = hps.epochs, hps.checkpoint_every, hps.print_every, args.reload_epoch, hps.l1_lambda
    model.train()
    losses, valid_losses = [], []
    best_valid_loss = 1e5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=hps.factor, patience=hps.patience, eps=1e-5, verbose=True)
    
    plt.ion()
    plt.figure(1)
    num = 0
    
    for epoch in range(reload_epoch + 1, reload_epoch + 1 + epochs):
#         model.train()
        total_loss = total_char_loss = total_action_loss = total_word_loss = total_ppl = 0
        for i, train_data in enumerate(train_loader):
#             sents, sent_lens, chars, actions, attributes, titles, title_len = train_data
#             sents, sent_lens, chars, actions= sents.cuda(), sent_lens.cuda(), chars.cuda(), actions.cuda()
#             attributes, titles, title_len = attributes.cuda(), titles.cuda(), title_len.cuda()
            
    
            sents, sent_lens, chars, actions, _, attributes, attr_lens, titles, title_len, masks, char_weights = train_data
            sents, sent_lens, chars, actions= sents.cuda(), sent_lens.cuda(), chars.cuda(), actions.cuda()
            titles, title_len, masks, char_weights = titles.cuda(), title_len.cuda(), masks.cuda(), char_weights.cuda()
            attributes = [attribute.cuda() for attribute in attributes]
            attr_lens = [attr_len.cuda() for attr_len in attr_lens]
            
            opt.zero_grad()
            loss, char_loss, action_loss, word_loss, ppl = model(sents, sent_lens, chars, actions, attributes, attr_lens, titles, title_len, masks, char_weights)
#             reg_loss = cal_reg_loss(model)
#             loss += l1_lambda * reg_loss
            
            
            loss.backward()
            clip_grad_norm_(model.parameters(), hps.clip)
            opt.step()

            
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_char_loss += char_loss.item()
            total_word_loss += word_loss.item()
            total_ppl += ppl.item()
            
            
            if (i + 1) % print_every == 0:
                
                plt.plot(num, total_loss / print_every,'.')
                plt.draw()
                plt.savefig(hps.test_path + 'train_loss.png')
                num += 1
                
                losses.append(total_loss / print_every)
#                 scheduler.step(total_loss / print_every)
                print('epoch {} batch {} avg_loss {} char_loss {} action_loss {} word_loss {} ppl {}'.\
                      format(epoch, i, total_loss / print_every, total_char_loss / print_every,
                             total_action_loss / print_every, total_word_loss / print_every, total_ppl / print_every), flush=True)
                total_loss = total_char_loss = total_action_loss = total_word_loss = total_ppl = 0
        
        if (epoch + 1) % checkpoint_every == 0:
            print('validing', flush=True)
            valid_loss = valid(model, valid_loader, idx2word, hps)
            valid_losses.append(valid_loss)
            scheduler.step(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), hps.test_path + hps.save_path)
            model.train()
#     plt.plot(losses)
#     plt.savefig(hps.test_path + 'train_loss.png')
#     plt.plot(valid_losses)
#     plt.savefig(hps.test_path + 'valid_loss.png')
            
# main
def main(hps):
    
    #pre_train
    if args.pre_train:
        word2idx, idx2word, verb2idx, idx2verb = make_vocab(hps)
#         word2idx, idx2word, verb2idx, idx2verb = load_vocab(hps)
#         mapping, vectors = load_glove(hps)
#         weights_matrix = make_pre_trained_word_embedding(mapping, vectors, word2idx.keys(), hps)
        hps = hps._replace(vocab_size=len(word2idx))
        hps = hps._replace(verb_vocab_size=len(verb2idx))
        hps = hps._replace(pre_train=True)
        
        print('parameters:')
        print(hps)
        
        train_loader, valid_loader, char_weights, action_weights = load_train_data(word2idx, verb2idx, hps)
        model = Model(char_weights, action_weights, hps)
#         model.load_state_dict(torch.load(hps.test_path + 'models/pre_train.model'))
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
        print('pre_training', flush=True)
        pre_train(model, optimizer, train_loader, valid_loader, idx2word, hps)
        
    #train
    elif args.train:
#         word2idx, idx2word = make_vocab(hps)
        word2idx, idx2word = load_vocab(hps)
        hps = hps._replace(vocab_size=len(word2idx))
        
            
        print('parameters:')
        print(hps)
        
        train_loader, valid_loader, char_weights, action_weights = load_train_data(word2idx, hps)
        model = Model(char_weights, action_weights, hps)
#         model.load_state_dict(torch.load(hps.test_path + 'models/best.model'))
        if args.reload:
            model.load_state_dict(torch.load(hps.test_path + hps.save_path.format(args.reload_epoch)))

        model.cuda()
#         model = nn.DataParallel(model,device_ids=[0])
        optimizer = optim.Adam(model.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
        print('training', flush=True)
        train(model, optimizer, train_loader, valid_loader, idx2word, hps)
    
    #test
    if args.test:
        print('testing', flush=True)
        word2idx, idx2word = load_vocab(hps)
        hps = hps._replace(vocab_size=len(word2idx))
        hps = hps._replace(test=True)
        model = Model([0] * hps.max_num_char, [0] * hps.vocab_size, hps)
        model.load_state_dict(torch.load(hps.test_path + hps.save_path))
        model.cuda()
        test_loader, anony2names = load_test_data(word2idx, hps)
        test(model, test_loader, idx2word, anony2names, hps)

if __name__ == '__main__':
    main(hps)