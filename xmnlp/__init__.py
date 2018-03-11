# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

"""MIT License
Copyright (c) 2018 Sean
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

__author__ = 'sean lee'
__version__ = '0.1.7'

import sys
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


from . import postag as postagger
from . import summary
from . import checker as chkr
from . import sentiment as stm
from . import pinyin as pinyiner

def get_text(txt):
    if isinstance(txt, str):
        if sys.version_info[0] == 2:
            txt = txt.decode('utf-8')
    return txt

def load_stopword(fpath):
    import os

    stopwords = []
    def get_file(path):
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                if len(dirs) == 0:
                    for f in files:
                        yield os.sep.join([root, f])
        else:
            yield path

    for fname in get_file(fpath):
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                line = get_text(line)
                stopwords.append(line)
    return stopwords

def set_userdict(userdict):
    postagger.set_userdict(userdict)

class XmNLP(object):

    def __init__(self, doc=None, stopword=False, *args, **kwargs):
        self.doc = get_text(doc)
        self.userdict = None
        self.stopword = []
        if stopword:
            from .config import path as C_PATH
            self.stopword = load_stopword(C_PATH.stopword['corpus']['stopword'])

    def set_userdict(self, fpath):
        self.userdict = fpath

    def set_stopword(self, fpath):
        stopword = load_stopword(fpath)
        self.stopword = list(set(self.stopword + stopword))


    def seg(self, txt=None, hmm=True):
        if self.userdict != None:
            postagger.set_userdict(self.userdict)
            self.userdict = None

        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return postagger.seg(txt, hmm)
        return postagger.seg(self.doc, hmm)

    def tag(self, txt=None, hmm=True):
        if self.userdict != None:
            postagger.set_userdict(self.userdict)
            self.userdict = None
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return postagger.tag(txt, hmm)
        return postagger.tag(self.doc, hmm)

    def pinyin(self, txt=None):
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return pinyiner.translate(txt)
        return pinyiner.translate(self.doc)

    def checker(self, txt=None):
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return chkr.check(txt)
        return chkr.check(self.doc)

    def keyword(self, txt=None, k=10,
        allowPOS=['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']):
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return summary.keyword(txt, k=k, stopword=self.stopword, allowPOS=allowPOS)
        return summary.keyword(self.doc, k=k, stopword=self.stopword, allowPOS=allowPOS)

    def keyphrase(self, txt=None, k=10):
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return summary.keyphrase(txt, k=k, stopword=self.stopword)
        return summary.keyphrase(self.doc, k=k, stopword=self.stopword)

    def sentiment(self, txt=None):
        if self.doc == None and txt == None:
            return None
        if txt != None:
            txt = get_text(txt)
            return stm.predict(txt, stopword=self.stopword)
        return stm.predict(self.doc, stopword=self.stopword)

    def tag_mean(self, tag):
        from .config.tag import tag_dict
        return tag_dict.get(tag, 'undefined !')

# quick to xmnlp
def seg(txt, hmm=True):
    txt = get_text(txt)
    return postagger.seg(txt, hmm)

def tag(txt, hmm=True):
    txt = get_text(txt)
    return postagger.tag(txt, hmm)

def keyword(txt, k=10, stopword=None, 
    allowPOS=['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']):
    
    stopwords = []
    if stopword != None:
        stopwords = load_stopword(stopword)
    txt = get_text(txt)
    return summary.keyword(txt, k=k, stopword=stopwords, allowPOS=allowPOS)

def keyphrase(txt, k=10, stopword=None):
    stopwords = []
    if stopword != None:
        stopwords = load_stopword(stopword)
    txt = get_text(txt)
    return summary.keyphrase(txt, k=k, stopword=stopwords)

def pinyin(txt):
    txt = get_text(txt)
    return pinyiner.translate(txt)

def checker(txt):
    txt = get_text(txt)
    return chkr.check(txt)

def sentiment(txt, stopword=None):
    stopwords = []
    if stopword != None:
        stopwords = load_stopword(stopword)
    txt = get_text(txt)
    return stm.predict(txt, stopword=stopwords)

def tag_mean(tag):
    from .config.tag import tag_dict
    return tag_dict.get(tag, 'undefined !')
