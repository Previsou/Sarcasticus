"""
    Module: irony.nlp
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-06-11
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    
    Load 2 hand-made NLP tools (Sentence  and Word Tokenizer) and a couple utlity functions.
        
    Export:
        tokenize: Tokenize (sentence then words) a text
        parse: Tokenize and format result in a Pandas DataFrame
        toNgram: convert a sequence into n-gram sequence
        upToNgram: same as toNgram but for multiple arities
        sanitize: clean a text, assuming it is plain text.
        buildTF: compute unigram frequencies of a corpus.
"""

import re
from pandas import DataFrame, concat
from collections import Counter

from data import load
from parallel import parallel_imap

__all__ = ["toNgram", "upToNgram", "sanitize", "tokenize", "parse", "buildTF"]

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#           UTILITY FUNCTIONS

_text_tag = re.compile(r"<(p|h[1-6]).*?>(.+?)</\1>", re.UNICODE+re.I+re.DOTALL)
_tag_cleaner = re.compile("<.+?>", re.UNICODE)
        
def sanitize(raw_text, HTML=False):
    """
    Clean a text: remove trailing spaces, try to resolve encoding errors and deal with HTML markup.
    
    Args:
        str:: raw_text
        boolean::HTML: if True, HTML markup is removed
    Return:
        Unicode:: sanitized text
    """
    
    if HTML:
        clean = ""
        for match in _text_tag.finditer(raw_text):
            subtext = match.group(2)
            subtext = _tag_cleaner.sub("", subtext)
            subtext = subtext.strip()
            clean += u" " + subtext.strip()
    else:
        clean = raw_text
        
    try:
        clean = unicode(clean)
    except:
        clean = unicode(clean, errors="replace")
    clean = clean.replace(u"\n", u" ")
    return clean.strip()


def toNgram(sequence, n, same_size=False):
    """
    Convert a sequence into n-grams sequence
    
    Args:
        sequence : iterable object
        n : arity of the n-gram
        same_size: If True, the returned sequence will be left-padded with None to have the same size as input sequence
                    Useful to store the input and result in the same DataFrame.
    
    Return:
        sequence of arity n n-grams
    """
    temp =  zip(*[sequence[i:] for i in range(n)])
    if same_size:
        temp = [None]*(n-1) + temp
    return temp

def upToNgram(sequence, n, same_size=False):
    """
    Convert a sequence into n-grams sequences (one for each ngram arity).
    
    Args:
        sequence : iterable object
        n : maximum arity of the n-gram
        same_size: If True, the returned sequence will be left-padded with None to have the same size as input sequence
                    Useful to store the input and result in the same DataFrame.
    
    Return:
        [sequence of arity i n-grams for i=1 to n]
    """
    return reduce(lambda x,y: x+y, [toNgram(sequence, i, same_size) for i in range(1, n+1)])
    
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#           CUSTOM TOKENIZER


token_pattern = u"|".join([re.escape(_smiley) for _smiley in load("wordlist/emoticon")]) #EMOTICON
token_pattern += u"|mr\.|mrs\.|inc\.|ltd\.|prof\.|gen\."  # ABBREVIATIONS
token_pattern += u"|(?<![0-9])[.](?![0-9])" 
token_pattern += u"|".join([re.escape(_punkt) for _punkt in load("wordlist/punctuation")]) # PUNCTUATION
token_pattern += u"|[^\s.?!,;:()\[\]{}]+" # GENERAL CASE
token_pattern = re.compile(token_pattern, re.IGNORECASE + re.UNICODE)

_sent_pattern = u"([?!]+" # PUNCTUATION PATTERNS
_sent_pattern += u"|(?<![0-9]|[.])[.](?![0-9]|[.]))"
_sent_pattern = re.compile(_sent_pattern, re.IGNORECASE + re.UNICODE)


def tokenize(raw_text, sentence=True):
    """
    Hand-made text tokenizer. Take into account emoticon and special
    punctuation patterns that serves as features for further analysis.
    
    Perform sentence tokenization first (if sentence is set to True) then custom word tokenization.
    """
    if sentence:
        sentences = []
        temp = _sent_pattern.split(raw_text)
        for i in range(0,len(temp)-1,2):
            sentences.append(temp[i]+temp[i+1])
        if len(temp[-1])!=0:
            sentences.append(temp[-1])
        
        temp = [token_pattern.findall(sentence) for sentence in sentences]
        return [tokens for tokens in temp if len(tokens)!=0]
    else:
        return token_pattern.findall(raw_text)
    
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def parse(text):
    """
    Tokenize a text and return result in a pandas DataFrame.
    
    Args:
        text:   raw text
        
    Return:
        DataFrame: 2-level index (sent_id, token_id)
    """
    tokens = [DataFrame(x, columns=["word"]) for x in tokenize(text,sentence=True)]

    for t in tokens:
        t.index.name = "token_id"
        
    return concat(tokens, keys=range(len(tokens)), names=["sent_id", "token_id"])
        
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
        
def buildTF(data, n_jobs):
    """
    Compute term frequencies, given a corpus.
    
    Args:
        Iter::data: inerable of text
        Int::n_jobs: number of processors to use
    Return:
        Counter:: term frequencies
    """
    
    def occurences(text):
        temp = Counter()
        for match in token_pattern.finditer(text):
            token = match.group(0).lower()
            if token[0]!=u"@":   # Filter out Twitter UserID
                temp[token] += 1
        return temp
        
    freqs = Counter()
    for _, counts in parallel_imap(occurences, data, n_jobs=n_jobs):
        freqs.update(counts)

    return freqs
