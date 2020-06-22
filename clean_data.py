# from eventmakerTryServer import eventMaker
import json
import nltk
from tqdm import tqdm
# from dataCleaning import parseLine
import argparse
import random

import gzip
import xml.etree.ElementTree as ET

import spacy
from benepar.spacy_plugin import BeneparComponent

from multiprocessing import Pool

from time import time

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(BeneparComponent("benepar_en2"))


class parseLine(object):

	def __init__(self):
		self.splittedCC = []
		self.takeSBAR = []
		self.removedPP = []
		self.removedPunc = []
		self.parsedSent = []

	def getTree(self,inp):
# 		maker = eventMaker(inp)
# 		out = maker.callStanford(inp)
# 		out = json.loads(out)
# 		parseTreeStr = out["sentences"][0]["parse"]
		
		
		doc = nlp(inp)
		sent = list(doc.sents)[0]
		parseTreeStr = sent._.parse_string
		res = parseTreeStr + ";" + inp
		self.parsedSent.append(res)

	def splitCC(self):#,line):
		# This is a function that can split CC conjunction between Ss.
		for line in self.parsedSent:
			if "<EOS>" in line:
				self.splittedCC.append(line)

			else:
				sec = line.split(");")
				str1 = sec[0]+")"
				str2 = sec[1]

				if "(CC and) (S " in str1:
					sent = []
					split = str1.split("(CC and) (S ")

					for i, clause in enumerate(split):
						subsent = []
						#print(clause)
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent
# 					self.splittedCC += sent

				elif "(CC but) (S " in str1:
					sent = []
					split = str1.split("(CC but) (S ")

					for i, clause in enumerate(split):
						subsent = []
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent
# 					self.splittedCC += sent

				elif "(CC or) (S " in str1:
					sent = []
					split = str1.split("(CC or) (S ")

					for i, clause in enumerate(split):
						subsent = []
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent
# 					self.splittedCC += sent

				else:
					res = str1+"@@@@"+str2
					self.splittedCC.append(res)


	def rmSBAR(self):
		# input should only be the parse tree
		for inp in self.splittedCC:
			if "<EOS>" in inp:
				self.takeSBAR.append(inp)
				continue

			inp, _ = inp.split("@@@@")
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
			SBAR = []
			SBARtree = []
			for token in rest:
				number = 0 - list(token).count(")")
				token = token[:number]
				output.append(token)
				#print(token)
			for oneSBAR in allSBAR:
				sbar = []
				for token in oneSBAR:
					number = 0 - list(token).count(")")
					token = token[:number]
					sbar.append(token)
				SBAR.append(" ".join(sbar))
			for oneSBARtree in allSBARtree:
				SBARtree.append(" ".join(oneSBARtree))

			senttree = " ".join(resttree)
			sent = " ".join(output)

# 			if SBAR != []:
# 				res = senttree+"@@@@"+sent
# 				self.takeSBAR.append(res)

# 				for i, sbar in enumerate(SBAR): #write down all the SBARs
# 					first = (sbar.split(" ")[0]).lower()
# 					#print(first)
# 					if "he" in first or "she" in first or "they" in first or "who" in first or "what" in first: #manipulate the first word in SBAR
# 						res = SBARtree[i]+"@@@@"+sbar
# 						self.takeSBAR.append(res)
# 					else:
# 						temp = sbar.split(" ")[1:]
# 						rmFirst = " ".join(temp)
# 						temptree = SBARtree[i].split(" ")[2:]
# 						rmtree = " ".join(temptree)
# 						res = rmtree+"@@@@"+rmFirst
# 						self.takeSBAR.append(res)



# 			else:
			res = senttree+"@@@@"+sent
			self.takeSBAR.append(res)

	def rmPP(self):
		for inp in self.takeSBAR:
			if "<EOS" in inp:
				self.removedPP.append(inp)
				continue
			inp = inp.split("@@@@")[0]

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
						if ifPP < 0:
							ifPP = 0
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
			self.removedPP.append(res)

	def rmPunc(self):
		for line in self.removedPP:
			if "<EOS>" in line:
				self.removedPunc.append(line)
				continue

			line = line.replace(",","")
			line = line.replace(".","")
			line = line.replace("'","")
			line = line.replace(":","")
			line = line.replace(";","")
			line = line.strip()
			if line == "":
				continue
			elif len(line.split(" ")) > 0 and line.split(" ")[-1].isalpha() == True:
				line = line+"."
			elif len(line.split(" ")) > 0 and line.split(" ")[-1].isalpha() == False:
				temp = line.split(" ")[:-1]
				line = " ".join(temp)+"."

			if len(line.split(" ")) > 0 and line.split(" ")[0][0].islower():
				line = line.split(" ")[0].title()+" "+" ".join(line.split(" ")[1:])
				#print(line)

			self.removedPunc.append(line)	
			#print(line)
			#f.write(line+"\n")

def rmPunc(s):
	s = s.replace('. ,', '.')
	s = s.replace(', .', '.')
	s = s.replace(', ,', ',')
    
	return s

def f(item):
    try:
        id, sents = item
        output = []
        for s in sents:
            parser = parseLine()
            parser.getTree(s)
            parser.splitCC()
            output.append(' '.join([inp.split("@@@@")[1] for inp in parser.splittedCC]))
        return id + '\t' + rmPunc(' '.join(output))
    except:
        print(i, flush=True)
        return None
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file", default='input.txt')
    parser.add_argument("--output", type=str, help="output file", default='output.txt')
    args = parser.parse_args()

    lines = [l for l in open(args.input, 'r').read().split('\n') if len(l) > 0]
    lines = random.sample(lines, 100)
    ids = [l.split('\t')[0] for l in lines]
#     inputs = [l.split('\t')[1] for l in lines]
    inputs = [l.split('\t')[2] for l in lines]
    length = len(inputs)
    print('length: {}'.format(length))
    
#     t1 = time()
#     outputs = []
#     for i in tqdm(range(length)):
#         try:
#             id, sents = ids[i], [s.text for s in nlp(inputs[i]).sents] 
#             output = []
#             for s in sents:
#                 parser = parseLine()
#                 parser.getTree(s)
#                 parser.splitCC()
#         #         print(parser.splittedCC)
# #                 parser.rmSBAR()
#         #         print(parser.takeSBAR)
# #                 parser.rmPP()
#         #         print(parser.removedPP)
# #                 parser.rmPunc()
#         #         print(parser.removedPunc)
# #                 output.append(' '.join([inp.split("@@@@")[1] for inp in parser.takeSBAR]))
# #                 output.append(' '.join(parser.removedPP))
#                 output.append(' '.join([inp.split("@@@@")[1] for inp in parser.splittedCC]))
# #                 assert len(parser.removedPP) == len(parser.takeSBAR)
#             outputs.append(id + '\t' + rmPunc(' '.join(output)))
#         except:
#             print(i, flush=True)
#     open(args.output, 'w', encoding='utf-8').write('\n'.join(outputs))
#     t2 = time()
#     print('time:{}'.format(t2 - t1))
    
    t1 = time()
    items = [(id, [s.text for s in nlp(input).sents]) for id, input in zip(ids, inputs)]
    outputs = None
    with Pool(5) as p:
        outputs = p.map(f, items)
    open(args.output, 'w', encoding='utf-8').write('\n'.join([o for o in outputs if o != None]))
    t2 = time()
    print('time:{}'.format(t2 - t1))