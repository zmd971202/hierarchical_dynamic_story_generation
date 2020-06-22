# from eventmakerTryServer import eventMaker
import json
import nltk
from tqdm import tqdm
# from dataCleaning import parseLine
import argparse
import random

import gzip
import xml.etree.ElementTree as ET

# import spacy
# from benepar.spacy_plugin import BeneparComponent
# nlp = spacy.load('en_core_web_lg')
# nlp.add_pipe(BeneparComponent("benepar_en2"))

import benepar
sent_parser = benepar.Parser("benepar_en2")

def rmSBAR(inp):
    # input should only be the parse tree
    
    tree = inp.split(" ")
    #print(tree)
    ifSBAR = 0
    SBARencountered = False 
    SBAR = []
    SBARtree = []
    allSBAR = []
    allSBARtree = []
    rest = []
    resttree = []
    for token in tree:
        if "(SBAR" in token and SBARencountered == False:
            ifSBAR += 1
            SBARencountered = True
            continue
        if ifSBAR != 0:
            SBARtree.append(token)
            if "(" in token:
                number = list(token).count("(")
                ifSBAR = ifSBAR + number
                #print(number)
            elif ")" in token:
                SBAR.append(token)
                #print(SBAR)
                number = list(token).count(")")
                ifSBAR = ifSBAR - number
                if ifSBAR <= 0: #<0 or <= 0?
                    allSBAR.append(SBAR)
                    allSBARtree.append(SBARtree)
                    ifSBAR = 0
                    SBARencountered = False
                    SBAR = []
                    SBARtree = []

        elif ifSBAR == 0:
            resttree.append(token)
            if ")" in token:
                rest.append(token)

    output = []
    for token in rest:
        number = 0 - list(token).count(")")
        token = token[:number]
        output.append(token)

    senttree = " ".join(resttree)
    sent = " ".join(output)
    
    return senttree, sent

def rmPP(inp):
    cleaned = []
    #print(inp)
    check = inp.split(" ")[0]
    if ")" in check:
        temp = inp.split(" ")[1:]
        inp = " ".join(temp)

    parseTreeStr = inp
    parseTree = parseTreeStr.split(" ")
    ifPP = 0
    PPencountered = False
    for token in parseTree:
        if "(PP" in token and PPencountered == False:
            ifPP += 1
            PPencountered = True
            continue
        if ifPP != 0:
            if "(" in token:
                number = list(token).count("(")
                ifPP = ifPP + number
                #print(number)
            elif ")" in token:
                number = list(token).count(")")
                ifPP = ifPP - number
# 						if ifPP < 0:
# 							ifPP = 0
                if ifPP <= 0:
                    ifPP = 0
                    PPencountered = False
            #print(ifPP)
        elif ifPP == 0:
            if ")" in token:
                cleaned.append(token)
        #print(ifPP)
    output = []
    for token in cleaned:
        number = 0 - list(token).count(")")
        token = token[:number]
        output.append(token)
        #print(token)
    res = " ".join(output)
    return res

def rmPunc(s):
	s = s.replace('. ,', '.')
	s = s.replace(', .', '.')
	s = s.replace(', ,', ',')
	return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file", default='input.txt')
    parser.add_argument("--output", type=str, help="output file", default='output.txt')
    args = parser.parse_args()

    lines = [l for l in open(args.input, 'r').read().split('\n') if len(l) > 0]
#     random.seed(1234)
#     lines = random.sample(lines, 100)
    ids = [l.split('\t')[0] for l in lines]
    inputs = [l.split('\t')[1] for l in lines]
    length = len(inputs)
    print('length: {}'.format(length))

    outputs = []
    for i in tqdm(range(length)):
        try:
            id, sents = ids[i], inputs[i].split(' ||| ')
            
        
            output = []
            for s in sents:
#                 parser = parseLine()
#                 parser.getTree(s)
#                 parser.splitCC()
        #         print(parser.splittedCC)
#                 parser.rmSBAR()
        #         print(parser.takeSBAR)
#                 parser.rmPP()
        #         print(parser.removedPP)
#                 parser.rmPunc()
        #         print(parser.removedPunc)
#                 output.append(' '.join([inp.split("@@@@")[1] for inp in parser.takeSBAR]))
#                 output.append(' '.join(parser.removedPP))
                tree = ' '.join(str(sent_parser.parse(s)).split('\n'))
#                 tree = list(nlp(s).sents)[0]._.parse_string
                tree, res = rmSBAR(tree)
#                 res = rmPP(tree)
                output.append(res)
#                 output.append(' '.join([inp.split("@@@@")[1] for inp in parser.splittedCC]))
#                 assert len(parser.removedPP) == len(parser.takeSBAR)
            assert len(output) == len(sents)
            outputs.append(id + '\t' + rmPunc(' ||| '.join(output)))
        except Exception as e:
            print(e)
            print(i)
    open(args.output, 'w', encoding='utf-8').write('\n'.join(outputs))