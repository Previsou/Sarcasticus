from data.preprocessing import LabelEncoder
from nlp import sanitize
from parallel import parallel_imap

from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from scipy import sparse
import numpy as np
import cPickle
import gc

from itertools import imap
from sklearn.feature_extraction import FeatureHasher
from nlp import token_pattern
import random

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

class Sarcalingua(object):

    def __init__(self, featureExtractor, model=SGDClassifier(alpha=1e-5, penalty="l1", loss="modified_huber")):
        
        self.featureExtractor = featureExtractor
        self.classifier = model
        self.outEncoder = LabelEncoder()
               
    def sanitize(self, raw_text, HTML=False):
        return sanitize(raw_text, HTML)           
               
    def extractFeatures(self, clean_text):
        
        data, ij = self.featureExtractor.extract(clean_text)
        return sparse.coo_matrix((data, ij), shape=(1, self.featureExtractor.cols), dtype=float).tocsr()
    
    def predict(self, featureVector, proba=False):
        if proba:
            return self.classifier.predict_proba(featureVector.todense())
        else:
            return self.classifier.predict(featureVector)[0]

    def classify(self, raw_text, proba=False, HTML=False):
        
        pred = self.predict(self.extractFeatures(self.sanitize(raw_text, HTML)), proba)
        if proba:
            return pred[0,1]
        else:
            return self.outEncoder.inverse[pred]
    
    def corpusToDataset(self, chunkIterator, column_label, HTML=False, n_jobs=-1):
        
        f = lambda raw_text: self.featureExtractor.extract(self.sanitize(raw_text, HTML))
        
        for chunk in chunkIterator:
            data, ij = [], [[],[]]
            for docID, temp in parallel_imap(f, chunk.text, n_jobs=n_jobs, timeout=1):
                data += temp[0]
                ij[0] += [docID for _ in temp[0]]
                ij[1] += temp[1][1]
            
            X = sparse.coo_matrix((data, ij), shape=(len(chunk), self.featureExtractor.cols), dtype=float).tocsr()
            y = np.array(self.outEncoder.fit_transform(chunk[column_label]))
            
            yield X,y
            gc.collect()
        
    def train(self, chunkIterator, column_label, **parameters):
        
        self.classifier.set_params(**parameters)
        n_jobs = parameters.get("n_jobs",None)
        if n_jobs!=None:
            del parameters["n_jobs"]
        else:
            n_jobs = -1
        
        for chunkID, (X,y) in enumerate(self.corpusToDataset(chunkIterator, column_label, n_jobs=n_jobs)):
            print "Chunk %d" % chunkID
            print X.__repr__()
            print y.sum()
            
            self.classifier.partial_fit(X, y, classes=np.arange(len(self.outEncoder)))
                
        return self

    def evaluate(self, chunkIterator, preprocessed=False, report="coarse", **parameters):
    
        y_pred, y_true = np.empty(0), np.empty(0)
        
        if preprocessed:
            dataIterator = chunkIterator
        else:
            column_label = parameters["column_label"]
            del parameters["column_label"]
            dataIterator = self.corpusToDataset(chunkIterator, column_label, **parameters)
            
        for chunkID, (X,y) in enumerate(dataIterator):
            y_pred = np.hstack([y_pred, self.classifier.predict(X)])
            y_true = np.hstack([y_true,y])
            print chunkID, X.__repr__()
        
        
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        accuracy = float(confusion_matrix.trace())/confusion_matrix.sum()

        summary = metrics.classification_report(y_true, y_pred, target_names=map(str,self.outEncoder.classes_)) + "\n"
        summary += "Accuracy :"+str(accuracy)+"\n\n"
        print summary
    
        if report=="fine":
            
            reg_label = self.outEncoder.labels["regular"]
            y_coarse_pred = y_pred != reg_label
            y_coarse_true = y_true != reg_label
            confusion_matrix = metrics.confusion_matrix(y_coarse_true, y_coarse_pred)
            accuracy = float(confusion_matrix.trace())/confusion_matrix.sum()

            summary2 = metrics.classification_report(y_coarse_true, y_coarse_pred, target_names=["regular", "virony"]) + "\n"
            summary2 += "Accuracy :"+str(accuracy)+"\n\n"
            print summary2
    


class HashSarca(Sarcalingua):
    
    def __init__(self, nbits=20, model=SGDClassifier(alpha=1e-5, penalty="l1", loss="modified_huber")):
        
        self.featureExtractor = FeatureHasher(pow(2,nbits), input_type="pair")
        self.classifier = model
        self.outEncoder = LabelEncoder()
        self.drop_outs = set((   u"#sarcasm", u"#sarcastic", u"#ironic", u"#irony",
                    u"#sarcasme", u"#sarcastique", u"#ironie", u"ironique",
                    u"uncyclopedia", u"wikipedia"))
        
    def extractFeatures(self, clean_text):
        return self.featureExtractor.transform( (token_pattern.finditer(clean_text),) )
        
    def corpusToDataset(self, chunkIterator, column_label, HTML=False, **args):
        
        def prepare(raw_text):
            tokens = token_pattern.findall(self.sanitize(raw_text, HTML))
            if random.random() < 0.5:   # we delete the drop-outs half the time
                tokens = [tok for tok in tokens if tok not in self.drop_outs]
            try:
                alpha = 1./len(tokens)  #1./(1+log(len(tokens)))
                return ((tok.lower(), alpha) for tok in tokens)
            except ZeroDivisionError:
                return tuple()
        
        for chunk in chunkIterator:
            X = self.featureExtractor.transform(imap(prepare, chunk.text))
            y = np.array(self.outEncoder.fit_transform(chunk[column_label]))
            
            yield X,y
            gc.collect()
        
