"""
    Script: getSpecTokens.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-07-23
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Simple script to extract the most k frequent tokens, for each text label of the corpus.
    Calling term_freqs.py before is necessary, as it builds the term frequnecy vectors.
    
    Args:
        str::category: corpus name
        str::lang: language of the corpus (stopwords are always included)
        int::top: keep top most frequent tokens, for each label + top most frequent tokens appearing inly in one label.
            If set to "all", no filtering is performed.
    Return:
        None
        special_tokens is saved under "wordlist/corpusName_special_tokens.pickle"
"""


from virony import data
from collections import Counter
import sys


category = sys.argv[1]
lang = sys.argv[2]
top = sys.argv[3]


freqs = data.load("TF/"+category+"_term_freqs.pickle")
    
    
special_tokens = set()

special_tokens.update(u".?!\"'()[]{},;:")
special_tokens.update(token for token in data.load("wordlist/punctuation"))
special_tokens.update(token for token in data.load("wordlist/stopword_"+lang).word)


if top=="all":
    special_tokens.update([unicode(w) for w in freqs["irony"].iterkeys()])
    special_tokens.update([unicode(w) for w in freqs["regular"].iterkeys()])
else:
    top = int(top)
    
    
    scale = float(freqs["irony"]["the"]) / freqs["regular"]["the"]

    irony_only = Counter({k:v for k,v in freqs["irony"].iteritems() if freqs["regular"][k]==0})
    regular_only = Counter({k:v for k,v in freqs["regular"].iteritems() if freqs["irony"][k]==0})

    odds_irony = Counter({k:v/(scale * freqs["regular"][k]) for k,v in freqs["irony"].iteritems() if 0. < freqs["regular"][k]})
    odds_regular = Counter({k:(v * scale / freqs["irony"][k]) for k,v in freqs["regular"].iteritems() if 0. < freqs["irony"][k]})


    special_tokens.update([unicode(w) for w,_ in irony_only.most_common(top)])
    special_tokens.update([unicode(w) for w,_ in regular_only.most_common(top)])
    special_tokens.update([unicode(w) for w,_ in odds_irony.most_common(top)])
    special_tokens.update([unicode(w) for w,_ in odds_regular.most_common(top)])

    
data.save(special_tokens, "wordlist/"+category+"_special_tokens.pickle")
