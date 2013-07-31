"""
    Script: onlineTrainModels.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-07-23
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Simple script to train a classifer using online learning. Evaluation
    is performed on another dataset (could be the same). No splitting of
    data is done here, If necesary, make it beforehand.
    
    Args:
        str::train_corpus: corpus to train the classifier on
        str::eval_corpus: corpus to evaluate the classifier on
        str::grain: "coarse"-> regular/virony classification, "fine"-> multiclass
        int::n_bits: number of unigram features (2^n_bits). If 0, then One Hot encoding is assumed.
        int::chunksize: size of the chunks of the large corpus
        int::n_jobs: number of processors to use
        
    Return:
        None
        save trained model under "model/CORPUS_NAME_GRAIN(_xBIT)"
        print evaluation report
"""

from virony.data import load, load_chunk, save
import sys

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

train_corpus = sys.argv[1]
eval_corpus = sys.argv[2]
grain = sys.argv[3]
n_bits = int(sys.argv[4])
chunksize = int(sys.argv[5])
n_jobs = int(sys.argv[6])

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
print(u"Loading feature extractor")
if n_bits==0:
    from virony.sarcalingua import Sarcalingua
    from virony.feature import TextFeatureExtractor
    from virony.data.preprocessing import LabelEncoder
    
    words = LabelEncoder(load("wordlist/"+train_corpus+"_special_tokens.pickle"))
    extractor = TextFeatureExtractor(words)
    #extractor_bow = TextFeatureExtractor(words, features=[])

    sark = Sarcalingua(extractor)
    #sark_bow = Sarcalingua(extractor_bow)
    
else:
    from virony.sarcalingua import HashSarca
    
    sark = HashSarca(n_bits)

sark.classifier.n_iter = 5

if grain=="fine":
    column_label = "label"
else:
    column_label = "irony"

sark = HashSarca(n_bits)
sark.classifier.n_iter = 5

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
print("Load corpus")
corpus = load_chunk("corpus/"+corpus_name+".csv", chunksize, n_iter=sark.classifier.n_iter)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
print(u"Training...")
sark.train(corpus, column_label, n_jobs=n_jobs, verbose=1)

print("Saving model")
if n_bits==0:
    save(sark, "model/"+corpus_name.upper()+"_"+grain.upper())
else:
    save(sark, "model/"+corpus_name.upper()+"_"+grain.upper()+"_"+str(n_bits)+"BIT")

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
print(u"Evaluation...")
sark.evaluate(load_chunk("corpus/"+eval_corpus+".csv", chunksize=chunksize), column_label=column_label, n_jobs=n_jobs)
