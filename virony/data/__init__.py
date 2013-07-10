"""
    Module: irony.data
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-05-13
    This module is part of the code I wrote for my master research project at Keio University (Japan).

    
    Export:
        _PATH_TO_DATA: default path to the different data objects used in the package
        get_ressources: list available ressources (e.g. get_ressources("corpus") will list available corpora)
        save: save an object into the ressource store (e.g save(foo, "corpus/bar"))
        load: load an objet of the ressource store (e.g. load("corpus/sarcasm"))
        
        
    __all__(modules):
        analyzer
        preprocessing
    
        
    For more information, see respective modules.
        
"""
from os import listdir
from pandas import load as pandas_load
from pandas import read_csv
import cPickle

_PATH_TO_DATA_LINUX = "/usr/home/maxime/hilbert_home/virony_data/"

listdir(_PATH_TO_DATA_LINUX)
_PATH_TO_DATA = _PATH_TO_DATA_LINUX


__all__ = ["save", "load", "load_chunk", "get_ressources",'_PATH_TO_DATA', 'preprocessing']


def get_ressources(group_name):
    
    return listdir(_PATH_TO_DATA+group_name)
    
    
def save(obj, path):
    
    try:
        obj.to_csv(_PATH_TO_DATA+path, encoding="utf-8")
    except:
        with open(_PATH_TO_DATA+path, "wb") as fout:
            cPickle.dump(obj, fout)
            
def load(path):
    
    try:
        with open(_PATH_TO_DATA+path, "rb") as fin:
            obj = cPickle.load(fin)
        if obj==None:
            return pandas_load(_PATH_TO_DATA+path)
        else:
            return obj
    except:
        return pandas_load(_PATH_TO_DATA+path)
        
def load_chunk(path, chunksize, n_iter=1):
    
    for _ in xrange(n_iter):
        for chunk in read_csv(_PATH_TO_DATA+path, header=0, index_col=0, encoding="utf-8", chunksize=chunksize):
            yield chunk
        
