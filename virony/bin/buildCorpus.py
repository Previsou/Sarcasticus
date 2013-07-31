"""
    Script: buildCorpus.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-05-13
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
    Build corpus resources I use for my research.
    Resulting DataFrames are saved in the ressource store under "corpus/"
"""

from pandas import DataFrame, Index
import re
import os
import codecs
import xml.dom.minidom as minidom
from itertools import chain, imap
from random import shuffle

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class Corpus(object):
    """
    Base class to load corpus. Do not use directly.
    """
    def __init__(self, path):
        
        self.corpus = dict()
        self.path = path
        self.load()
    
    def load(self):
        pass
        
    def toDataFrame(self, *indexes):
        
        if len(indexes)==0:
            indexes = self.corpus.keys()
        
        entries = []
        for index in indexes:
                entries.append(self.corpus[index])
        
        return DataFrame(entries, index=Index(indexes, name="review"))
            
#----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------- 
    
class SarcasmCorpus(Corpus):
    """
    Load a sarcastic corpus.
    For more information on this corpus, see:
    Elena Filatova. Irony and Sarcasm: Corpus Generation and Analysis Using Crowdsourcing.
    """
    
    def __init__(self, path):
        
        self.corpus = dict()
        self.pair = re.compile(r"""PAIR:\t(?P<f1>[0-9A-Z_]+)[a-z()\t]*(?P<f2>[0-9A-Z_]+)""")
        self.single = re.compile(r"""(?P<label>[A-Z]+):\t(?P<f>[0-9A-Z_]+)""")
        self.review = re.compile(r"""<STARS> (?P<rating>[0-9.]+) </STARS>\n
                            <TITLE> (?P<title>.+) </TITLE>\n
                            .*
                            <PRODUCT> (?P<product>.+) </PRODUCT>\n
                            <REVIEW> (?P<text>.+) </REVIEW>""", re.X + re.DOTALL)
        self.path = path
        self.load()
    
    def load(self):
        
        with codecs.open(self.path+"file_pairing.txt",mode="r", encoding='utf-8') as fin:
            for line in fin.readlines():
                m = self.pair.search(line)
                if m != None:
                    m = m.groupdict()
                    
                    self.corpus[m["f1"]] = {"paired_with": m["f2"],
                                            "label": "sarcasm",
                                            "coarse": "verbal irony"}           
                    self.corpus[m["f2"]] = {"paired_with": m["f1"],
                                            "label": "regular",
                                            "coarse":"regular"}
                                                
                    self._loadReview(m["f1"], "ironic")
                    self._loadReview(m["f2"], "regular")
                    
                else:
                    m = self.single.search(line).groupdict()
                    if m["label"]=="REGULAR":
                        label, coarse = "regular", "regular"
                        toLoad = "regular"
                    else:
                        label, coarse = "sarcasm", "verbal irony"
                        toLoad = "ironic"
                        
                    self.corpus[m["f"]] = {"label": label, "coarse":coarse}
                    
                    self._loadReview(m["f"], toLoad)
            
    def _loadReview(self, fileid, label):
        
        with codecs.open(self.path+label+"/"+fileid+".txt", "r") as review:
            infos = self.review.search(review.read()).groupdict()
            infos["rating"] = float(infos["rating"])
            infos["text"] = infos["title"] + "\n" + infos["text"]   # keep the title in the document boost classification accuracy
            self.corpus[fileid].update(infos)
                
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
    
class SatireCorpus(Corpus):
    """
    Load a newswire corpus, labeled with satirical newswire.
    For more information on this corpus, see:
    Burfoot, Clint and Baldwin, Timothy (2009). Automatic Satire Detection : Are You Having a Laugh ?

    """

    def __init__(self, path):
        
        self.corpus = {"docs":[], "ids": []}
        self.path = path
        self.load()
    
    def load(self):
        
        with codecs.open(self.path+"training-class", encoding='iso-8859-15') as fin:
            infos = [line.split(u' ') for line in fin.readlines()]  
        with codecs.open(self.path+'test-class', encoding='iso-8859-15') as fin:
            infos += [line.split(u' ') for line in fin.readlines()]
        
        self.corpus["docs"] =  [doc for doc in imap(self.format, infos) if doc['text']!='']
        self.corpus["ids"] = map(lambda x: x[0], infos)
        
    def format(self,(fileid, label)):
        if fileid.find('test')==-1:
            type = 'training'
        else:
            type = 'test'
            
        if label[:-1] == "true":
            fine, coarse = "regular", "regular"
        else:
            fine, coarse = "satire", "verbal irony"
            
        with codecs.open(self.path+type+'/'+fileid, encoding='iso-8859-15') as newswiref:
            content = map(lambda x: x.strip(), newswiref.readlines())
            title = content[0]
            content = filter(lambda line: len(line)>0, content[2:])
            content = u' '.join(content)
        return {"title": title, "text": title+"\n"+content, "label": fine, "coarse":coarse, 'type':type}


    def toDataFrame(self):
        
        return DataFrame(self.corpus["docs"], index=self.corpus["ids"])
        
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

#### see directly reviewExtractor in raw_corpora            ####
class AmazonCorpus(Corpus):
    """
    Load the Amazon corpus into a DataFrame.
    This corpus is taken from: 
    Reyes, Antonio and Rosso, Paolo. Mining Subjective Knowledge from Customer Reviews : A Specific Case of Irony Detection. Computational Linguistics No. 1975 (2011)
    """
    def __init__(self, path):
    
        self.corpus = []
        self.path = path
        self.load()
      
    def load(self):
        import cPickle
        with open(self.path+'amazon_irony.pkl', 'rb') as fin:
            self.corpus = cPickle.load(fin)
                
    def toDataFrame(self):
        
        return DataFrame(self.corpus)

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class MultiDomain(Corpus):
    """
    Load MultiDomain corpus.
    For more information on this corpus, see: Multi-Domain Sentiment Dataset (standard dataset)
    """
    def __init__(self, path):
        
        self.path = path
        self.corpus = []
        self.load()
        
        
    def toDataFrame(self):
        
        return DataFrame(self.corpus,)

    def loadFile(self,path):

        with open(path) as fin:
    
            xml = minidom.parse(fin)
            reviews_obj = xml.getElementsByTagName(u'review')

            attributes = []
            for review_obj in reviews_obj:
                children = review_obj.childNodes
                for child in children:
                    if child.nodeType != child.TEXT_NODE:
                        attributes.append(child)

                
                document = {"label":"regular", "coarse":"regular"}
                for attribute in attributes:
                    for node in attribute.childNodes:
                        if attribute.localName=='review_text':
                            document['text'] = node.data
                        else:
                            document[attribute.localName] = node.data
                document["text"] = document["title"] + "\n" + document["text"]

                # Clean up
                for key, value in document.items():
                    cleaned = u''.join(value.split('\n'))
                    try:
                        document[key] = int(cleaned)
                    except:
                        try:
                            document[key] = float(cleaned)
                        except:
                            document[key] = cleaned
                
                self.corpus.append(document)

    def load(self):
        
        # Extract admissible directories
        directories = [self.path+directory for directory in os.listdir(self.path) if directory[0]!='.']
        directories = filter(os.path.isdir, directories)
        
        for directory in directories:
            for label in ['positive', 'negative']:
                path = directory+'/'+label+'.review'
                print path
                self.loadFile(path)
                
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

if __name__== '__main__':

    from virony import data
    from pandas import concat
    
    _PATH = '/usr/home/maxime/hilbert_home/raw_corpora/'
    
    print(u'Parsing Amazon corpus')
    amazon = AmazonCorpus(_PATH+'amazon/').toDataFrame()
    
    print(u'Parsing Multidomain corpus')
    multiDomain = MultiDomain(_PATH+'multi_domain/').toDataFrame()
    amazon = concat([amazon, multiDomain], ignore_index=True)
    data.save(amazon,"corpus/amazon")
    
    print(u'Parsing sarcasm corpus')
    sarcasm = SarcasmCorpus(_PATH+'SarcasmCorpus/', ).toDataFrame()
    data.save(sarcasm, "corpus/sarcasm")
    
    
    print(u'Parsing satire corpus')
    satire = SatireCorpus(_PATH+'satire/').toDataFrame()
    data.save(satire, "corpus/satire")
