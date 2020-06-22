# from eventmakerTryServer import eventMaker
import json
import nltk
from tqdm import tqdm
# from dataCleaning import parseLine
import argparse
import random

from helper import *

from multiprocessing import Pool

import spacy
nlp = spacy.load('en_core_web_lg')

def f(input):
    id, summary, char, attr = input
    sents = summary.split(' ||| ')

    output = []
    num_sent = no_char = no_action = 0
    for sent in sents:
        num_sent += 1

        character = '<noc>'
        action = '<unk>'

        s = list(nlp(sent).sents)[0]
#         try:
#             s = list(nlp(sent).sents)[0]
#         except:
# #                 print(id)
# #                 print(sent)
# #                 exit(0)
#             no_char += 1
#             output.append(character + ' ; ' + action)

        candidates = []
        svos = findSVOs(s)
        for svo in svos:
            rec, flag = '', False
            for c in char:
                if svo[0] in c:
                    rec = c
                    flag = True
                    break
            if flag:
                if svo[1] == s.root.text:#.lemma_:
                    character, action = rec, svo[1]
                    break
                else:
                    candidates.append((rec, svo[1]))
        if candidates != [] and character == '<noc>':
            character, action = random.choice(candidates)
            output.append(character + ' ; ' + action)
            continue


        nsubjs = None
        try:
            root = s.root
            action = root.text #lemma_
            nsubjs = [c.text for c in root.children if c.dep_ == 'nsubj']
        except:
            action = '<unk>'
            no_action += 1


        candidates = []
        for c in char:
            if c in sent:
                candidates.append(c)
        if candidates == []:
            no_char += 1
        else:
            if nsubjs == None:
                character = random.choice(candidates)
            else:
                for candidate in candidates:
                    if any([nsubj in candidate for nsubj in nsubjs]):
                        character = candidate
                        break
                if character == '<noc>':
                    character = random.choice(candidates)

        output.append(character + ' ; ' + action)
    
#     return (id + '\t' + ' ||| '.join(output), num_sent, no_char, no_action)
    return (id + '\t' + ' ; '.join(attr) + '\t' + ' ||| '.join(output), num_sent, no_char, no_action)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent", type=str, help="sentence file", default='sent.txt')
    parser.add_argument("--char", type=str, help="character file", default='char.txt')
    parser.add_argument("--attr", type=str, help="attribute file", default='attribute.txt')
    parser.add_argument("--output", type=str, help="output file", default='output.txt')
    args = parser.parse_args()
    
    max_num_char = 10

    lines = [l for l in open(args.char, 'r').read().split('\n') if len(l) > 0]
    id2attr = {l.split('\t')[0]:l.split('\t')[1] for l in open(args.attr, 'r').read().split('\n') if len(l) > 0}
    chars = {}
    attrs = {}
    for l in lines:
        id, cs = l.split('\t')
        try:
            attr = id2attr[id]
        except:
            continue
        chars[id] = cs.split(' ; ')[:max_num_char]
        attrs[id] = attr.split(' ; ')[:max_num_char]
    
    lines = [l for l in open(args.sent, 'r').read().split('\n') if len(l) > 0]
    summaries = {l.split('\t')[0]:l.split('\t')[1] for l in lines}
    ids = [id for id in summaries]
    

#     random.seed(2468)
#     ids = random.sample(ids, 100)
    
    length = len(ids)
    print('length: {}'.format(length))

    
#     outputs = []
#     num_sent = no_char = no_action = 0
#     for i in tqdm(range(length)):
#         id = ids[i]
#         summary = summaries[id]
#         try:
#             char = chars[id]
#             attr = attrs[id]
#         except:
#             continue
#         sents = summary.split(' ||| ')

#         output = []
#         for sent in sents:
#             num_sent += 1
            
#             character = '<NOC>'
#             action = '<UNK>'
            
#             try:
#                 s = list(nlp(sent).sents)[0]
#             except:
# #                 print(id)
# #                 print(sent)
# #                 exit(0)
#                 no_char += 1
#                 output.append(character + ' ; ' + action)
                
#             candidates = []
#             svos = findSVOs(s)
#             for svo in svos:
#                 rec, flag = '', False
#                 for c in char:
#                     if svo[0] in c:
#                         rec = c
#                         flag = True
#                         break
#                 if flag:
#                     if svo[1] == s.root.lemma_:
#                         character, action = rec, svo[1]
#                         break
#                     else:
#                         candidates.append((rec, svo[1]))
#             if candidates != [] and character == '<NOC>':
#                 character, action = random.choice(candidates)
#                 output.append(character + ' ; ' + action)
#                 continue
            
            
#             nsubjs = None
#             try:
#                 root = s.root
#                 action = root.lemma_
#                 nsubjs = [c.text for c in root.children if c.dep_ == 'nsubj']
#             except:
#                 action = '<UNK>'
#                 no_action += 1
            
            
#             candidates = []
#             for c in char:
#                 if c in sent:
# #                     character = c
# #                     break
#                     candidates.append(c)
#             if candidates == []:
#                 no_char += 1
#             else:
#                 if nsubjs == None:
#                     character = random.choice(candidates)
#                 else:
#                     for candidate in candidates:
#                         if any([nsubj in candidate for nsubj in nsubjs]):
#                             character = candidate
#                             break
#                     if character == '<NOC>':
#                         character = random.choice(candidates)
                
#             output.append(character + ' ; ' + action)

#         outputs.append(id + '\t' + ' ; '.join(attr) + '\t' + ' ||| '.join(output))
    
#     print('no char: {}, no action: {}'.format(no_char / num_sent, no_action / num_sent))
#     open(args.output, 'w', encoding='utf-8').write('\n'.join(outputs))
    
#     random.seed(1456)
#     ids = random.sample(ids, 100)
    inputs = []
    for id in ids:
        summary = summaries[id]
        try:
            char = chars[id]
            attr = attrs[id]
        except:
            continue
        inputs.append((id, summary, char, attr))
    outs = None
    with Pool(5) as p:
        outs = p.map(f, inputs)
    
    strings = []
    total_num_sent = total_no_char = total_no_action = 0
    for o in outs:
        strings.append(o[0])
        total_num_sent += o[1]
        total_no_char += o[2]
        total_no_action += o[3]
    print('no char: {}, no action: {}'.format(total_no_char / total_num_sent, total_no_action / total_num_sent))
    open(args.output, 'w', encoding='utf-8').write('\n'.join(strings))