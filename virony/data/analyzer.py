"""
    Module: irony.data.analyzer
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2013-05-13
    This module is part of the code I wrote for my master research project at Keio University (Japan).
    
    DEPRECATED DOCUMENTATION

    This module provides basic feature analysis.
    All functions in this module uses Pandas DataFrame, features are assumed to
    correspond to a column of a DataFrame.
    
    
    Export:
        analyze1D: analysis of 1 feature (coverage, box plot)
        analyze2D: analyze 1 feature against another (correlation, box-plot)
"""

#######   MIGHT BE PROBLEMS WITH SUBPLOTS (if yes, pass ax objects)

from __init__ import load

from pandas import DataFrame, concat
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates, scatter_matrix
from scipy import stats

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               Utility function

def getFeatureDataFrame(dataset_name):

    features = [ "capitalized", "looong", "emoticon", "human",
            "society", "slang", "polarity"]
    stats = ["mean", "std"]
    offset = len(features) * len(stats)

    X,y,y_encoder = load("dataset/"+dataset_name)
    X = X[:,:offset].todense()

    complexF = DataFrame(X, columns=[feature+"_"+stat for feature in features for stat in stats])
    complexF["irony"] = y_encoder.inverse_transform(y)
    
    return complexF

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#               Feature graphs


def analyze1D(dataset, column_label="irony", out="DISPLAY"):
    """
    Perform the analysis on features, one at a time.
    
    Args:
        dataset: DataFrame instance
        out(string): ""DISPLAY" will plot the results, a filepath will save figures to the location pointed by out
        box(boolean): If True, plot box-plot of the features over labels
    Return:
        None
    """
    print('Plot saved in '+out)
    

    for i,feature in enumerate([f for f in dataset.columns if f!=column_label]):
        
        try:
            # Box  plot
            dataset.boxplot(column=[feature], by=[column_label])
            if out=='DISPLAY':
                plt.show()
            else:
                plt.savefig(out+feature+"_box")
            plt.clf()
        except:
            print(u"Cannot make "+feature+" box plot")

        try:
            # KDE plot
            for label, group in dataset.groupby(column_label):
                group[feature].plot(kind="kde", label=label, title=feature)
            plt.legend(loc="best")
        
            if out=='DISPLAY':
                plt.show()
            else:
                plt.savefig(out+feature+"_kde")
            plt.clf()
        except:
            print(u"Cannot make "+feature+" kde plot")
        
def scatterPlot(dataset, out="DISPLAY"):
    graph = scatter_matrix(dataset)
    if out=='DISPLAY':
        plt.show()
    else:
        plt.savefig(out+"scatter_matrix")
    plt.clf()
    return graph

def parallelCoordinates(dataset, column_label="irony", out="DISPLAY"):
    graph =  parallel_coordinates(dataset, column_label, alpha=0.5)
    if out=='DISPLAY':
        plt.show()
    else:
        plt.savefig(out+"parallel_coordinates")
    plt.clf()
    return graph
    
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#           Statistical tests on features

def KS_test(dataset, column_label="irony"):
    
    groups = {label:g for label, g in dataset.groupby(column_label)}
    key0, key1 = groups.keys()[:2]
    results = {}
    for feature in dataset.columns:
        try:
            sample0 = groups[key0][feature]
            sample1 = groups[key1][feature]
            results[feature] = stats.ks_2samp(sample0,sample1)
        except:
            results[feature] = ("error", "erorr")
        
    return results
        
def Welch_test(dataset, column_label="irony"):
    
    groups = {label:g for label, g in dataset.groupby(column_label)}
    key0, key1 = groups.keys()[:2]
    results = {}
    for feature in dataset.columns:
        try:
            sample0 = groups[key0][feature]
            sample1 = groups[key1][feature]
            results[feature] = stats.ttest_ind(sample0, sample1, equal_var=False)
        except:
            results[feature] = ("error", "erorr")
    return results

def correlation(dataset, column_label="irony"):
    
    corrs = {"global":dataset.corr()}
    for name, group in dataset.groupby(column_label):
        corrs[column_label+"_"+str(name)] = group.corr()
    return corrs
                
def mean_intervals(dataset, column_label="irony", alpha=0.05):
    
    intervals = []
    for feature in dataset.columns:
        if feature!=column_label:
            temp = {}
            for label, group in dataset.groupby(column_label):
                mean, (inf, sup) = stats.bayes_mvs(group[feature], alpha=1-alpha)[0]
                temp[label] = {"inf":inf, "mean":mean, "sup":sup}
            intervals.append(DataFrame(temp).T)
    
    return concat(intervals, keys=dataset.columns)
            
