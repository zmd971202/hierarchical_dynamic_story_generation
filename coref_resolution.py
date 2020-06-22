import spacy
import neuralcoref
import random
import argparse
from tqdm import tqdm

from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="input file", default='input.txt')
parser.add_argument("--output", type=str, help="output file", default='output.txt')
parser.add_argument("--sent_output", type=str, help="output file", default='sent_output.txt')
args = parser.parse_args()

nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)

lines = open(args.input, 'r').read().split('\n')
# lines = random.sample(lines, 100)
length = len(lines)
print(length)

def opt_sent(words, corefs, start):
    keys = sorted(corefs.keys())
    output, rec = [], 0
    
    for key in keys:
        output += words[rec:key - start]
        end, main = corefs[key]
        output.append(main)
        rec = end - start
    output += words[rec:len(words)]
    return ' '.join(output)

def clean_sents(sents, words):
    new_sents, new_words = [], []
    flag = False
    for s, ws in zip(sents, words):
        if s == '"' and not flag:
            if len(new_sents) != 0:
                new_sents[-1] += '"'
                new_words[-1].append('"')
                flag = True
        else:
            new_sents.append(s)
            new_words.append(ws)
            flag = False
    return new_sents, new_words
    
# all_outputs = []
# all_sents = []
# for i in tqdm(range(length)):
# #     if i % 1000 == 0:
# #         print(i, flush=True)
#     l = lines[i]
#     sid, l = l.split('\t')
#     doc = nlp(l)
#     sents = [s.text.strip() for s in doc.sents]
#     words = [[w.text for w in s] for s in doc.sents]
#     sents, words = clean_sents(sents, words)
# #     doc = nlp(' '.join(sents))
#     sent2id = {s:i for i, s in enumerate(sents)}
# #     words = [[w.text for w in s] for s in doc.sents]
#     corefs = [{} for _ in range(len(words))]
# #     lengths = [len(s) for s in words]
    
#     for cluster in doc._.coref_clusters:
#         main = cluster.main.text
#         for mention in cluster.mentions[1:]:
#             start, end = mention.start, mention.end
#             try:
#                 id = sent2id[mention.sent.text.strip()]
#             except:
#                 id = sent2id[mention.sent.text.strip() + '"']
# #                 t = mention.sent.text
# #                 for s in sent2id:
# #                     if 
#             corefs[id][start] = (end, main)
# #     print(corefs)
#     outputs, start = [], 0
#     for i, s in enumerate(words):
#         outputs.append(opt_sent(s, corefs[i], start))
#         start += len(s)
# #     print(sid + '\t' + ' ||| '.join(outputs))
#     all_outputs.append(sid + '\t' + ' ||| '.join(outputs))
#     all_sents.append(sid + '\t' + ' ||| '.join([' '.join(ws) for ws in words]))


def f(l):
    sid, l = l.split('\t')
    doc = nlp(l)
    sents = [s.text.strip() for s in doc.sents]
    words = [[w.text for w in s] for s in doc.sents]
    sents, words = clean_sents(sents, words)

    sent2id = {s:i for i, s in enumerate(sents)}

    corefs = [{} for _ in range(len(words))]
    
    for cluster in doc._.coref_clusters:
        main = cluster.main.text
        for mention in cluster.mentions[1:]:
            start, end = mention.start, mention.end
            try:
                id = sent2id[mention.sent.text.strip()]
            except:
                id = sent2id[mention.sent.text.strip() + '"']

            corefs[id][start] = (end, main)

    outputs, start = [], 0
    for i, s in enumerate(words):
        outputs.append(opt_sent(s, corefs[i], start))
        start += len(s)
    
    return (sid + '\t' + ' ||| '.join(outputs), sid + '\t' + ' ||| '.join([' '.join(ws) for ws in words]))
    
with Pool(5) as p:
    outs = p.map(f, lines)
all_outputs, all_sents = [], []
for o in outs:
    all_outputs.append(o[0])
    all_sents.append(o[1])
open(args.output, 'w', encoding='utf-8').write('\n'.join(all_outputs))
open(args.sent_output, 'w', encoding='utf-8').write('\n'.join(all_sents))