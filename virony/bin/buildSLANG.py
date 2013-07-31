"""
    Script: buildSLANG.py
    Author: Lubin Maxime <maxime@soft.ics.keio.ac.jp>
    Date: 2012-12-11
    This module is part of the code I wrote for my master research project at Keio University (Japan)
    
        ################################################################
        ##   Problem with accessing onlineslangdictionary (TO DEBUG)  ##
        ################################################################
        
    Build the slang wordlist resource by scraping the Online Slang Dicitonary (http://onlineslangdictionary.com)
    
    All the data is stored in a Pandas DataFrame:
     - 2 columns: known, vulgarity (scores between 0. and 1.)
     - index: slang phrases
     
    Result is saved in the ressource store under "wordlist/slang".
    
"""
import add_to_path
import urllib2
import re
import codecs
from pandas import DataFrame

from virony.parallel import parallel_map
from virony import data


base_url = "http://onlineslangdictionary.com"

list_urls = ["/word-list/0-a/",
                "/word-list/b-b/",
                "/word-list/c-d/",
                "/word-list/e-f/",
                "/word-list/g-g/",
                "/word-list/h-k/",
                "/word-list/l-o/",
                "/word-list/p-r/",
                "/word-list/s-s/",
                "/word-list/t-t/",
                "/word-list/u-w/",
                "/word-list/x-z/"]
                

definition_pattern = re.compile("""<a href="(?P<url>/meaning-definition-of/.+?)">(?P<expression>.+?)</a>""")
heard_pattern = re.compile("""var vote_count_heard = (?P<count_heard>[0-9]+);.*?var vote_count_never = (?P<count_never>[0-9]+);""", re.DOTALL)
vulgarity_pattern = re.compile("""<span id="vulgarity-vote-average">(?P<vulgarity>[0-9]{1,3})%</span>""")



def getSlangInfos((url, expression)):
    try:
        print expression
        raw = urllib2.urlopen(base_url+url).read().encode("utf8")
        known = map(float, heard_pattern.search(raw).groups())
        try:
            known = known[0] / sum(known)
        except:
            known = 0.

        return expression ,{"vulgarity": float(vulgarity_pattern.search(raw).group("vulgarity"))/100,
                "known": known}
    except:
        return None
            
            
slang = []
for wordlist in list_urls[:1]:
    url = base_url+wordlist
    print url
    raw = urllib2.urlopen(url).read().encode("utf8")
    def_urls = definition_pattern.findall(raw)[:10]
    
    slang += parallel_map(getSlangInfos, def_urls, n_jobs=4)

slang = DataFrame(map(lambda x: x[1], slang), index=map(lambda x: x[0], slang))
slang = slang[slang.known > 0]  # remove slang expressions that nobody knows :)

data.save(slang, "wordlist/slang")

