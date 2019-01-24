# -*- coding: utf-8 -*-
import io
import json
import readability
from nltk.tokenize import word_tokenize
import pandas

f = io.open("data/songs/der_erlkonig.txt", mode="r", encoding="utf-8")
text = f.read()
# print(" ".join(word_tokenize(text)))

data = readability.getmeasures(text, lang='de')
# print(json.dumps(data, indent=4))

freq_filename = "data/DeReKo-2014-II-MainArchive-STT.100000.freq/DeReKo-2014-II-MainArchive-STT.100000.freq"

import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
import string
punctuations = list(set(string.punctuation) | set(['<','>','«', '»']))
stop_words = set(stopwords.words('german'))
columns=['token', 'lemma', 'speech_part', 'frequency']

# https://www.fluentu.com/blog/how-many-words-do-i-need-to-know/

# Functional beginner: 250-500 words. After just a week or so of learning, you’ll already have most of the 
# tools to start having basic, everyday conversations. In most of the world’s languages, 500 words will be 
# more than enough to get you through any tourist situations and everyday introductions.

# Conversational: 1,000-3,000 words. With around 1,000 words in most languages, you’ll be able to ask people 
# how they’re doing, tell them about your day and navigate everyday life situations like shopping and public transit.

# Advanced: 4,000-10,000 words. As you grow past the 3,000 word mark or so in most languages, you’re moving 
# beyond the words that make up everyday conversation and into specialized vocabulary for talking about your 
# professional field, news and current events, opinions and more complex, abstract verbal feats. At this point, 
# you should be able to reach C2 level in the Common European Framework for Reference (CEFR) in most languages.

# Fluent: 10,000+ words. At around 10,000 words in many languages, you’ve reached a near-native level of vocabulary, 
# with the requisite words for talking about nearly any topic in detail. Furthermore, you recognize enough words 
# in every utterance that you usually understand the unfamiliar ones from context.

# Native: 10,000-30,000+ words. Total word counts vary widely between world languages, making it difficult to say how 
# many words native speakers know in general. As we discussed above, estimates of how many words are known by the 
# average native English speaker vary from 10,000 to 65,000+.

fluency = {
	"beginner": (250,500),
	"conversational": (1000, 3000),
	"advanced": (4000, 10000),
	"fluent": (10000, 10000),
	"native": (10000, 30000),
}
freq_df = pd.DataFrame(columns=columns)

limit = fluency['conversational'][1]
lines = []
with open(freq_filename) as f:
    for line in f:
    	formatted_line = line.lower().strip().split('\t')
    	token = formatted_line[0]
    	if token in stop_words:
    		continue
    	if any(punc in token for punc in punctuations):
    		continue
    	if any(char.isdigit() for char in token):
    		continue
    	lemma = formatted_line[1]
    	frequency = formatted_line[3]
    	if (freq_df['lemma'] == lemma).any():
    		freq_df.loc[freq_df['lemma'] == lemma, 'frequency'] += frequency
    	else:
    		freq_df.loc[len(freq_df)] = formatted_line
    	
    	if len(freq_df) >= limit:
    		break

freq_df.sort_values(by=['frequency'])
print(freq_df.loc[0:fluency['beginner'][1]])

