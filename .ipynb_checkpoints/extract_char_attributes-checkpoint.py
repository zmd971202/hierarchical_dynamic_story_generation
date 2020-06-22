# for spacy 2.1.0
from collections import defaultdict
import argparse
import spacy
import random
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--summary", type=str, help="summary", default='summary.txt')
parser.add_argument("--char", type=str, help="char", default='char.txt')
parser.add_argument("--output", type=str, help="output file", default='output.txt')
args = parser.parse_args()

nlp = spacy.load("en_core_web_lg")


def extract_label(doc, token2char, id2char):
    des = defaultdict(lambda:set())
    for i, token in enumerate(doc):
        if i not in token2char:
            continue
        char_id = token2char[i]
        char = id2char[char_id]

        if token.dep_ in ("nsubj", "nsubjpass", "attr", "pobj", "dobj", "poss", "appos", "compound", "conj"):
            head = token.head

            outMap = set()
            """check head"""
            head_dep = head.dep_
            while head_dep not in ("advcl", "ROOT", "pcomp", "xcomp", "ccomp"):
                head = head.head
                head_dep = head.dep_

            outMap.add(head.lemma_)
            for child in token.children:
                child_dep = child.dep_
                child_tag = child.tag_

                if 'NN' in child_tag and child.text not in char:
                    for c in child.children:
                        if 'JJ' in c.tag_:
                            outMap.add(c.lemma_)
                    outMap.add(child.lemma_)
                if 'JJ' in child_tag:
                    outMap.add(child.lemma_)
                elif 'VB' in child_tag:
                    outMap.add(child.lemma_)

            """create des"""
            if len(outMap):
                des[char_id] = des[char_id].union(set(outMap))
    return des

lines = [l for l in open(args.summary, 'r').read().split('\n') if len(l) > 0]

characters = open(args.char, 'r').read().split('\n')
id2chars = {}
for character in characters:
    id, cs = character.split('\t')
    id2chars[id] = cs.split(' ; ')

# lines = random.sample(lines, 100)

# length = len(lines)
# strings = []
# for j in tqdm(range(length)):
#     l = lines[j]
#     id, summary = l.split('\t')
#     summary = ' '.join(summary.split(' ||| '))
    
#     doc = nlp(summary)
    
#     try:
#         chars = id2chars[id]
#     except:
#         continue
    
#     char2id = {char:i for i, char in enumerate(chars)}
#     id2char = {i:char for i, char in enumerate(chars)}
# #     print(char2id)
#     token2char = {}
    
#     for ent in doc.ents:
#         w, pos = ent.text.strip(), ent.label_
#         if w.endswith("'s") or w.endswith("’s"):
#             w = w[:-2].strip()
# #         print(w)
#         if pos == 'PERSON' and w in chars:
# #             print(w)
#             start, end, idx = ent.start, ent.end, char2id[w]
#             for i in range(start, end):
#                 token2char[i] = idx
    
# #     print(id)
# #     print(token2char)
#     des = extract_label(doc, token2char, id2char)
    
# #     print(id)
# #     for i, char in enumerate(chars):
# #         print(i, char)
# #         print(','.join(list(des[char2id[char]])))
    
#     strings.append(id + '\t' + ' ; '.join([char + '@@@' + ' '.join([w for w in list(des[char2id[char]]) if w != ';']) for char in chars]))

def f(input):
    l, chars = input
    id, summary = l.split('\t')
    summary = ' '.join(summary.split(' ||| '))
    
    doc = nlp(summary)
    
    char2id = {char:i for i, char in enumerate(chars)}
    id2char = {i:char for i, char in enumerate(chars)}

    token2char = {}
    
    for ent in doc.ents:
        w, pos = ent.text.strip(), ent.label_
        if w.endswith("'s") or w.endswith("’s"):
            w = w[:-2].strip()

        if pos == 'PERSON' and w in chars:

            start, end, idx = ent.start, ent.end, char2id[w]
            for i in range(start, end):
                token2char[i] = idx
    
    des = extract_label(doc, token2char, id2char)
    
    return id + '\t' + ' ; '.join([char + '@@@' + ' '.join([w for w in list(des[char2id[char]]) if w != ';']) for char in chars])

# lines = random.sample(lines, 100)
inputs = []
for l in lines:
    id, summary = l.split('\t')
    try:
        chars = id2chars[id]
        inputs.append((l, chars))
    except:
        continue
with Pool(5) as p:
    strings = p.map(f, inputs)
open(args.output, 'w', encoding='utf-8').write('\n'.join(strings))