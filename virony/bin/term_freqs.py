"""
    Script: term_freqs.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-07-22
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Simple scrip to extract term occurences in a cirpus.
    
    Args:
        category: name of the corpus to analyze
        chunk_size: how many rows at a time
        n_jobs: number of processors to use
        
    Return:
        None
        Results are saved under "corpus/category".
"""


import add_to_path
from virony.nlp import buildTF
from virony import data
from pandas import read_csv
from collections import Counter
import sys


category = sys.argv[1]
chunk_size = int(sys.argv[2])
n_jobs = int(sys.argv[3])


print(u"Loading corpus")
corpus = read_csv(data._PATH_TO_DATA+"corpus/"+category+".csv", chunksize=chunk_size, encoding="utf-8")

print(u"Building Term Frequencies")
freqs = {"irony":Counter(), "regular":Counter()}

for chunk in corpus:
    print(u"GROUP::IRONY")
    freqs["irony"] += buildTF(chunk[chunk.irony].text, n_jobs)

    print(u"GROUP::REGULAR")
    freqs["regular"] += buildTF(chunk[chunk.irony==False].text, n_jobs)
    
print(u"Saving Term Frequencies")
data.save(freqs, "TF/"+category+"_term_freqs.pickle")
