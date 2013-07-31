"""
    Script: feature.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-07-23
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
   Module regrouping basic feature extraction utilities and definitions.
   It also provides a simple feature extractor.
   
   Export:
        features: a default list of features [(name, callable)]
        allCaps, looong, emoticon, slang, polarity, human, society: feature functions
        TextFeatureExtractor: basic feature extraction class
"""

from data import load, _PATH_TO_DATA
from nlp import token_pattern
from data.preprocessing import LabelEncoder

from math import sqrt
import random
import cPickle
import re

__all__ = ["allCaps", "looong", "emoticon", "slang", "polarity", "human",
                "society", "features", "TextFeatureExtractor"]


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#                   PROSODIC FEATURES

def allCaps(entry):
    """ 
    Return True if a word is capitalized, with the exception of "I", "A"
        
    Args:
        entry: dictionary (or Series) (entry["word"] need to exist)
    Return:
        boolean
    """
    
    word = entry["word"]
    return word.isalpha() and word.upper()==word and word!=u"I" and word!="A"  #ad hoc rules
    

_repete = re.compile(r"(\w)\1\1+", re.UNICODE)
def looong(entry):
    """ 
    Return True if a word is lengthened, e.g. "Riiiight"
        
    Args:
        entry: dictionary (or Series) (entry["word"] need to exist)
    Return:
        boolean
    """
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
    """
    Return the vulgarity score (0 to 1) of the given entry.
    Scores are taken from the Online Slang Dictionary.
    
    Args:
        entry: dictionary or Series, entry["normalized"] should exist
    
    Return:
        float::vulgarity
    """
    
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
    """
    Return True if entry["normalized"] is in "wordlist/human"
    """
    return entry["normalized"] in _humanOriented_data

def society(entry):
    """
    Return True if entry["normalized"] is in "wordlist/society"
    """
    return entry["normalized"] in _societyOriented_data
     
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               EMOTIONAL FEATURES

# Extracted averaged polarity of a lemma, given SentiWordNet scores 
_SENTIWN = load("wordlist/polarities.pkl")

def polarity(entry):
    """
    Return the polarity of a given entry, according to SentiWordNet.
    
    Args:
        entry: diuctionary or Series, entry["normalized"] should be possible.
    """
    return _SENTIWN.get(entry["normalized"], 0.)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               PRAGMATIC FEATURES

_EMOTICON = set(load("wordlist/emoticon").values)

def emoticon(entry):
    return entry["word"] in _EMOTICON


#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Default list fo features to consider for feature extraction.
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

# Set of words set out during feature extraction, because they abnormally appear due to corpus gathering method.
drop_outs = set((   u"#sarcasm", u"#sarcastic", u"#ironic", u"#irony",
                    u"#sarcasme", u"#sarcastique", u"#ironie", u"ironique",
                    u"uncyclopedia", u"wikipedia"))


class TextFeatureExtractor(object):
    """
    Simple feature extractor. Perform tokenization, feature extraction,
    compute mean and std of features over a document, extract unigram frequency features,
    and return sparse vectors.
    """
    
    def __init__(self, wordEncoder, features=features):
        """
        Initialize the feature extractor.
        
        Args:
            data.preprocessing.LabelEncoder::wordEncoder: OneHot encoder
            [(str,callable)]::features: special features to extract.
        Return:
            None
        """
        
        self.features = features
                            
        self.stats = 2  # number of statisitcs to copute (here mean and std)
        self.offset = self.stats * len(self.features)
         
        self.encoder = wordEncoder
        
    # total number of features (used to build sparse matrix)
    cols = property(lambda x: x.offset + len(x.encoder))
        

    def extract(self, raw_text):
        """
        Perform feature extraction. The return of this function can be piped
        to scipy sparse constructors.
        
        Args:
            str::raw_text
        Return:
            list::data: feaure values
            (list,list)::ij: (rows, cols) of the feature values.
        """
        data, ij = [], [[],[]] 
        
        # Keep track of statistics while iterating over tokens
        stats = {feature[0]:{"mean":0., "std":0.} for feature in self.features} 
        N = 0.
        
        for match in token_pattern.finditer(raw_text):
            N += 1
            
            # Use for retro-compatibility with feature functions (see virony.feature)
            token = {"word":match.group(0), "normalized":match.group(0).lower()}
            
            # Complex features
            for feat in self.features:
                temp = feat[1](token)
                stats[feat[0]]["mean"] += temp
                stats[feat[0]]["std"] += temp**2
            
            # Words in drop_outs are specially processed: they are discarded
            # altogether randomly, to prevent classification to rely too heavily on those
            # while saving part of their information/predicitve power.
            keep = True
            if token["normalized"] in drop_outs:
                keep = random.random() > 0.5
            
            if keep:
                # Unigram frequency features (normalization by total number of tokens is performed below)
                col = self.encoder.labels.get(token["normalized"], -1)
                    
                if col!=-1:
                    data.append(1.0)      # Duplicate entries will be summed together when converting to CSR
                    ij[0].append(0)
                    ij[1].append(col+self.offset)
          
        if N>0:
            # Normalize unigram frequency features
            data = [val/N for val in data]
            
            # Finish computing statistics: mean and std
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
        
    
