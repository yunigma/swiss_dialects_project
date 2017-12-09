# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

#################### 
threshold = 180 
minimum = 12 
minhits = 5 
filename = 'vsm.csv'
featureDB = 'featureDB.csv'
#################### end of settings

dialects = ['BE','BS','ZH','LU']
classid = {'BE': 1, 'BS': 2, 'ZH': 3, 'LU': 4, 'Label': 0}

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from collections import defaultdict
import logging
import argparse
import random
import codecs
import sys
from typing import Set, Tuple, Iterable
import numpy as np
import re
from sklearn.decomposition import PCA
sep = []
freq = {}
alltrigrams = []
set(alltrigrams)
sentences = []
sentclass = []


featureDBcols = "feature,type,max,conf,mean,BE,BS,ZH,LU\n"
settings = str(classid) + "," + "threshold:" + str(threshold) + ",minimum:" + str(minimum) + ",minhits:" + str(minhits) + "\n"

file = open(filename, 'w')
file.write(settings)
file.close()
fdb = open(featureDB, 'w')
fdb.write(featureDBcols)
fdb.close()
file = open(filename, 'a')
fdb = open(featureDB, 'a')

print("Now opening training file 'train.csv'")
print("Using settings:", settings)

for line in open("train.csv"):
    line = line.rstrip() 
    
    sep  = re.findall("(.+)\,(.+)", line)
    line2 = list(sep[0])
    #print("Line:", line2)
    if (len(line2) > 1):             
        words = line2[0]
        sentences.append(words)
        cat = line2[1]
        sentclass.append(cat)
        if (cat != 'Label'):
            wordlist = re.split(" ", words)
            for w in wordlist:
                le = len(w)
                
                trigrams = re.findall("(.{3})", w)
                loop = [0,1]
                for i in loop:
                    w2 = w[1:]
                    w = w2
                    trig = re.findall("(.{3})", w)
                    if (trig):
                        for j in trig:
                            trigrams.append(j) 
                for t in trigrams:
                    trigramcat = t + "," + cat
                    try:
                        freq[trigramcat] = int(freq[trigramcat]) + 1
                    except:
                        freq[trigramcat] = 1
                    if t not in alltrigrams:
                        alltrigrams.append(t)
                        
sentcount = len(sentences)
print("Sentences found:",sentcount)


print("Dialects:",dialects)
featurecount = len(alltrigrams)
selfeatures = []
for t in alltrigrams:
    dcount = {}
    fline = ""   
    allcounts = "" 
    for d in dialects:
        trigramcat = t + "," + d
        try:
            f = freq[trigramcat]
        except:
            f = 0
        dcount[d] = f
        allcounts = allcounts + "," + str(f)
        
    sum = 0
    max = 0 
    maxcat = "" 
    for k,v in dcount.items():
        sum = sum + v
        if (v > max): 
            max = v 
            maxcat = k 
    mean = sum / 4
    
    hits = 0
    maxdiff = 0
    for k,v in dcount.items():
        diff = int(abs(v - mean)/mean*100) 
        if (diff >= threshold): 
            if (v >= minimum):   
                hits = hits + 1
                if (diff > maxdiff):  
                    maxdiff = diff  
    if (hits):
        selfeatures.append(t) 
        fline = t + ",trigram," + maxcat + "," + str(maxdiff) + "," + str(mean) + allcounts + "\n"
        fdb.write(fline) 

print("Number of features found:", featurecount)
selcount = len(selfeatures)
i = 0
fnum = {}
for s in selfeatures:  
    fnum[s] = i
    i = i + 1
print(">>> selected for VSM:", selcount)
outline = "CLASS,"
outline = outline + ','.join(selfeatures) + '\n'
file.write(outline)

i = 0

vsm = np.zeros((selcount, sentcount)) 
print("Building the VSM, with max. dimensions:",vsm.shape, "features / sentences")
sehits = 0 
for s in sentences:
    outline = ""
    sentencehits = []
    hits = 0
    senlen = len(s)
    
    for f in selfeatures:
        hitscore = len(re.findall(f,s)) #how many times we can find the feature in that sentence
        if (len(re.findall(f,s)) > 0):
            hits = hits + 1

        sentencehits.append(str(hitscore))
    if (hits >= minhits):
        sep = ","
        id = classid[sentclass[i]]
        id = str(id)
        outline = id + sep + sep.join(sentencehits)
        sehits = sehits + 1
    i = i + 1 
    rest = i % 500
    if (rest == 0):
        print("   ", i, "of", sentcount, "sentences checked. Selected sentences:", sehits)
    if (outline):
        outline = outline + "\n"
        file.write(outline)
file.close()
fdb.close() 
print("Totally number of sentences incorporated into VSM output:", sehits)
print("Outputfile:", filename)
print("Stats for selected features in:",featureDB)







    
