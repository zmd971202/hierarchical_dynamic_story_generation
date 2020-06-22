# import en_core_web_lg
import spacy
import nltk
import random
from collections import defaultdict

# nlp = en_core_web_lg.load()
nlp = spacy.load("en_core_web_lg")

# cmu movie
# metadatas = open('MovieSummaries/movie.metadata.tsv', 'r').read().split('\n')[:-1]
# id2title = {l.split('\t')[0]:l.split('\t')[2] for l in metadatas}

# lines = open('MovieSummaries/plot_summaries.txt', 'r').read().split('\n')[:-1]


# wikiplots
id2title = {}
lines = []
for l in open('preprocessed/WikiPlots/data.txt', 'r').read().split('\n'):
    id, t, s = l.split('\t')
    id2title[id] = t
    lines.append(id + '\t' + s)





# lines = [lines[6844]]
# lines = random.sample(lines, 100)
strings = []
for i, l in enumerate(lines):
    if i % 5000 == 0:
        print(i, flush=True)
    
    id, summary = l.split('\t', 1)
    try:
        title = id2title[id]
#         sents = nltk.sent_tokenize(summary)
        doc = nlp(summary)
        chars = defaultdict(lambda:0) #set()
        
        for ent in doc.ents:
            w, pos = ent.text.strip(), ent.label_
            if pos == 'PERSON':
                if w.endswith("'s") or w.endswith("’s"):
                    w = w[:-2].strip()
                
                flag = False
                for char in chars:
                    if w in char:
                        flag = True
                        chars[char] += 1
                if flag:
                    continue
                
                chars[w] = 1
#         if len(chars) < 2:
#             print(id)
#         else:
        
        if len(list(chars.keys())) == 0:
            continue
    
        chars = sorted(list(chars.keys()),key=lambda x: chars[x],reverse=True)
        strings.append(id + '\t' + ' ; '.join(chars))
    except Exception as e:
#         print(e)
        print(id)
# print(strings)
# open('preprocessed/MovieSummaries/refine/id_with_char2.txt', 'w', encoding='utf-8').write('\n'.join(strings))
open('preprocessed/WikiPlots/id_with_char.txt', 'w', encoding='utf-8').write('\n'.join(strings))