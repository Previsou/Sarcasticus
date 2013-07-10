from data import load, _PATH_TO_DATA
from nlp import token_pattern
from data.preprocessing import LabelEncoder

from math import sqrt
import random
import cPickle
import re


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#                   PROSODIC FEATURES

def allCaps(entry):
    word = entry["word"]
    return word.isalpha() and word.upper()==word and word!=u"I"  #ad hoc rules
    

_repete = re.compile(r"(\w)\1\1+", re.UNICODE)
def looong(entry):
        
    word = entry["word"]
    
    #check if a word has been lenghtend i.e "riiiight"
    splitted = _repete.split(word)
    return len(splitted) > 1

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               VOCABULARY FEATURES

_SLANG = load("wordlist/slang")
_STOPWORD = set(load("wordlist/stopword_en").word.values)

def slang(entry):
    
    word = entry["normalized"]
    if word[0].isalpha() and "*" in word:   # e.g. Fu**, sh*thole, etc (all thos patterns are considered highly vulgar)
        return 0.6  # Arbitrary value (but around the values of "asshole" "fuck" etc)
    elif word not in _STOPWORD:
        try:
            return _SLANG.ix[word, "vulgarity"]
        except KeyError:
            return 0.
    else:
        return 0.

_humanOriented_data = set(load("wordlist/human").word.values)
_societyOriented_data = set(load("wordlist/society").word.values)

def human(entry):
    return entry["normalized"] in _humanOriented_data

def society(entry):
    return entry["normalized"] in _societyOriented_data
     
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               EMOTIONAL FEATURES

_SENTIWN = load("wordlist/polarities.pkl")


def polarity(entry):
    return _SENTIWN.get(entry["normalized"], 0.)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               PRAGMATIC FEATURES

_EMOTICON = set(load("wordlist/emoticon").values)

def emoticon(entry):
    return entry["word"] in _EMOTICON


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

features = [("capitalized"      ,   allCaps),
            ("looong"           ,   looong),
            ("emoticon"         ,   emoticon),
            ("human"            ,   human),
            ("society"          ,   society),
            ("slang"            ,   slang),
            ("polarity"         ,   polarity),
            ]
            
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

drop_outs = set((   u"#sarcasm", u"#sarcastic", u"#ironic", u"#irony",
                    u"#sarcasme", u"#sarcastique", u"#ironie", u"ironique",
                    u"uncyclopedia", u"wikipedia"))


class TextFeatureExtractor(object):
    
    def __init__(self, wordEncoder, features=features):
        
        self.features = features
                            
        self.stats = 2
        self.offset = self.stats * len(self.features)
         
        self.encoder = wordEncoder
    
    cols = property(lambda x: x.offset + len(x.encoder)) # total number of features (used to build sparse matrix (see prep_data.py))
        

    def extract(self, raw_text):
    
        data, ij = [], [[],[]] 
        stats = {feature[0]:{"mean":0., "std":0.} for feature in self.features} 
        N = 0.
        for match in token_pattern.finditer(raw_text):
            N += 1
            token = {"word":match.group(0), "normalized":match.group(0).lower()}
            
            # Complex features
            for feat in self.features:
                temp = feat[1](token)
                stats[feat[0]]["mean"] += temp
                stats[feat[0]]["std"] += temp**2
            
            keep = True
            if token["normalized"] in drop_outs:
                keep = random.random() > 0.5
            
            if keep:
                # Bag of words
                col = self.encoder.labels.get(token["normalized"], -1)
                    
                if col!=-1:
                    data.append(1.0)      # Duplicate entries will be summed together when converting to CSR
                    ij[0].append(0)
                    ij[1].append(col+self.offset)
          
        if N>0:
            # Normalize
            data = [val/N for val in data]
            
            for pos, (feature,_) in enumerate(self.features):
                mean = stats[feature]["mean"] / N
                data.append(mean)
                ij[0].append(0)
                ij[1].append(self.stats * pos)
                
                try:
                    biased_std = sqrt(stats[feature]["std"] / N - mean**2)
                    data.append(biased_std)
                    ij[0].append(0)
                    ij[1].append(self.stats * pos + 1)
                    
                except ValueError, e:
                    print e
                    print stats[feature]["std"] / N - mean**2
    
        return data, ij
        
    
