"""
    Script: buildWordnetList.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-05-013
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Build several WordNet-based lists of word I use as features for classification.
    Resulting lists are save into the ressource store under "wordlist/human" and "wordlist/society"
"""



from nltk.corpus import wordnet as wn
import codecs
from pandas import Series

from virony import data


# Wordnet version 3.0

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

def getLemmas(synsets, rec, acc=[]):
        
    if rec==0:
        acc += [lemma for syn in synsets for lemma in syn.lemma_names]
        return acc
    else:
        acc += [lemma for syn in synsets for lemma in syn.lemma_names]
        nexts = set(hypo for syn in synsets for hypo in syn.hyponyms())
        return getLemmas(nexts, rec-1, acc)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Building human centeredness (as done in Mining Subjective Knowledge from Customer Reviews: A Specific Case of Irony Detection, Antonio Reyes and Paolo Rosso)
# Retrieve all lemmas belonging to hyponyms of "relative", "relation", "relationship" and "kinship" synsets.
# More precisely: relative.n.01, all synsets of "relationship", relation.n.06, kinship.n.02

recursion_level = 7 #Maximum level for these synsets is 7 (418 lemmas)

humans = ["relative", "kin", "kinship", "relation", "relationship"]
humans += getLemmas([wn.synset('relative.n.01')], recursion_level)
#humans += getLemmas([wn.synset('relation.n.06')], recursion_level)     #included when recurse
#humans += getLemmas([wn.synset('kinship.n.02')], recursion_level)      # included when recurse
humans += getLemmas(wn.synsets('relationship'), rec=recursion_level)

humans = list(set(humans))
humans.sort()

print "Human oriented", len(humans)
human = Series(humans, name="humanOriented")
data.save(human, "wordlist/human")

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# Society Centeredness


# Extract all lemmas of "profession.n.01", profession.n.02", "occupation.n.01", "employment.n.02", politics (all) and their hyponyms

recursion_level = 5

society = ["society", "study", "work"]
society += getLemmas([wn.synset('profession.n.01')], recursion_level)
society += getLemmas([wn.synset('profession.n.02')], recursion_level)
society += getLemmas([wn.synset('occupation.n.01')], recursion_level)
society += getLemmas([wn.synset('employment.n.02')], recursion_level)
society += getLemmas(wn.synsets('politics'), recursion_level)

society = list(set(society))
society.sort()

print "Society oriented", len(society)
save(society, "societyOriented", SAVE_PATH)

