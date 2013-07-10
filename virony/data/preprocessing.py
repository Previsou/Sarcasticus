"""
    Module: irony.data.preprocessing
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-05-13
    This module is part of the code I wrote for my master research project at Keio University (Japan).
    
    This module provides facilities to manage datasets and their associated labels, as many supervised
    learning algorithm accept only integer labels.
    

    Export:
        standardize: standardize a DataFrame (center and scale features)
        LabelEncoder: Convert hashable labels into integer labels (and inverse mapping)
        LabelBinarizer: Implement One-Hot encodgin of hashable labels (and the inverse mapping)
"""

from numpy import zeros

__all__ = ["LabelEncoder", "LabelBinarizer"]


def standardize(dataset):
    """
    Standardize features. Each column (feature) will be centered and normalized.
    
    Args:
        dataset: DataFrame (one feature per column)
    Return:
        standardized DataFrame
    """
    return dataset.sub(dataset.mean(0), axis=1).div(dataset.std(0), axis=1)


class LabelEncoder(object):
    """
    Utility class.
    Convert hashabel labels into integer labels, for Machine Learning purposes.
    
    See also Label Binarizer.
    """
    def __init__(self, sequence=[]):
        self.labels = dict()
        self.inverse = dict()
        self.nb_labels = 0
        
        try:
            self.fit(sequence)
        except:
            pass

    classes_ = property(lambda x: [x.inverse[i] for i in range(x.nb_labels)] )
    
    def __len__(self):
        return self.nb_labels

    def fit(self, sequence):
        """
        Learn the mapping hashable labels <-> integer labels
        
        Args:
            sequence: an iterable of hashable objects
            
        PS: sequence should contain all labels appearing in the dataset
        """
        for item in sequence:
            if self.labels.get(item ,-1)==-1:
                self.labels[item] = self.nb_labels
                self.inverse[self.nb_labels] = item
                self.nb_labels +=1
                
        return self
        
    def transform(self, sequence):
        """
        Transform the given sequence of hashable labels into integer labels.
        
        Args:
            sequence: an iterable object
        """
        return map(lambda x: self.labels.get(x, -1), sequence)
        
    def inverse_transform(self, sequence):
        """
        Inverse mapping: integers -> hashable labels
        
        Args:
            sequence: an iterable object
        """
        return map(lambda x: self.inverse.get(x, None), sequence)
        
    def fit_transform(self, sequence):
        """
        Learn the mapping and perform transformation of the given sequence.
        Equivalent to: encoder.fit(sequence).transform(sequence)
        
        Args:
            sequence: an iterable object
        """
        return self.fit(sequence).transform(sequence)
    
    
class LabelBinarizer(LabelEncoder):
    """
    Implement One-Hot encoding of hashable labels.
    Only for small set of labels (non-sparse implementation).
    
    See also: irony.data.dataset.LabelEncoder
    """
    
    def transform(self, sequence):
        """
        Transform the given sequence of hashable labels into integer labels.
        
        Args:
            sequence: an iterable object
        """
        return map(self._binarize, super(LabelBinarizer, self).transform(sequence))
        
    def _binarize(self, int_label):
        """
        Convert an integer label into the One-Hot encoded vector.
        """
        temp = zeros(self.nb_labels, dtype=float)
        try:
            if int_label != -1:
                temp[int_label] = 1.
            return temp
        except:
            print int_label
            return temp
        
    def inverse_transform(self, sequence):
        """
        Inverse mapping: integers -> hashable labels
        
        Args:
            sequence: an iterable object
        """
        temp = map(self._unbinarize, sequence)
        return super(LabelBinarizer, self).inverse_transform(temp)
        
    def _unbinarize(self, array):
        """
        Convert a vector into an integer label.
        """
        try:
            i=0
            while array[i]==0.:
                i+=1
            return i
        except:
            return -1
        
