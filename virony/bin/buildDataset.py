"""
    Script: buildDataset.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-07-23
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Simple script to transofrm the corpus into Machine Learning representation: 
        - sparse matrix [n_docs, n_features]
        - target vector [n_doc]
        - target encoder: index/revert index to convert target code/labels.
    
    Args:
        str::corpus_name
        str::grain: "coarse" for binary (regular/virony) classification. "fine" for multiclass (regular/irony/sarcasm/satire)
        int::n_bits: number of hashed features. If 0, OneHot encoding is assumed.
    Return:
        None
        dataset is saved under "dataset/corpus_name_grain(_xbit)"
"""


from virony.sarcalingua import Sarcalingua, HashSarca
from virony.data import load, save, load_chunk
from virony.feature import TextFeatureExtractor
from virony.data.preprocessing import LabelEncoder
import numpy as np
from scipy import sparse
import sys

corpus_name = sys.argv[1]
grain = sys.argv[2]
n_bits = int(sys.argv[3])

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#   Load feature extractor

if n_bits==0:
    words = LabelEncoder(load("wordlist/"+corpus_name+"_special_tokens.pickle"))
    words = LabelEncoder(load("wordlist/tweetdata_special_tokens.pickle"))
    extractor = TextFeatureExtractor(words)

    sark = Sarcalingua(extractor)

else:
    sark = HashSarca(n_bits)


if grain=="fine":
    column_label = "label"
else:
    column_label = "irony"
    
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#   Load corpus

print(u"Load corpus")
corpus = load_chunk("corpus/"+corpus_name+".csv", chunksize=50000, n_iter=1)    

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#   Feature extraction

print(u"Extract features")

X,y = [], []
for Xsub, ysub in sark.corpusToDataset(corpus, column_label):
    X.append(Xsub)
    y.append(ysub)
    
X = sparse.vstack(X).tocsr()
y = np.hstack(y)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Saving

print(u"Save dataset")
if n_bits==0:
    save((X,y,sark.outEncoder), "dataset/"+corpus_name+"_"+grain)
else:
    save((X,y,sark.outEncoder), "dataset/"+corpus_name+"_"+grain+"_"+str(n_bits)+"bit")
    
