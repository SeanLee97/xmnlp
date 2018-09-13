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
__version__ = '0.1.8'

import os
import sys
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

from .config import path as C_PATH
from .config.tag import tag_dict
from . import postag as _postag
from . import summary as _summary
from . import checker as _checker
from . import sentiment as _sentiment
from . import pinyin as _pinyin
from . import radical as _radical
from .utils import safe_input, filelist

import io

def load_stopword(fpath):
    stopwords = set()
    for fname in filelist(fpath):
        with io.open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = safe_input(line)
                stopwords.add(line)
    return stopwords

sys_stopwords = load_stopword(C_PATH.stopword['corpus']['stopword'])

def set_stopword(fpath):
    global sys_stopwords
    words = load_stopword(fpath)
    sys_stopwords = set(words) | sys_stopwords

def set_userdict(fpath):
    _postag.set_userdict(fpath)

def seg(txt, hmm=True):
    txt = safe_input(txt)
    return _postag.seg(txt, hmm)

def tag(txt, hmm=True):
    txt = safe_input(txt)
    return _postag.tag(txt, hmm)

def keyword(txt, k=10, stopword=True, 
    allowPOS=['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']):
    global sys_stopwords

    txt = safe_input(txt)
    if stopword:
        stopwords = sys_stopwords
    else:
        stopwords = []
    
    return _summary.keyword(txt, k=k, stopword=stopwords, allowPOS=allowPOS)

def keyphrase(txt, k=10, stopword=True):
    global sys_stopwords

    txt = safe_input(txt)
    if stopword:
        stopwords = sys_stopwords
    else:
        stopwords = []
    return _summary.keyphrase(txt, k=k, stopword=stopwords)

def pinyin(txt):
    txt = safe_input(txt)
    return _pinyin.translate(txt)

"""
Args:
    level:
        - 0: word
        - 1: doc
"""
def checker(txt, level=0):
    txt = safe_input(txt)
    return _checker.check(txt, level=level)

def sentiment(txt, stopword=True):
    global sys_stopwords

    txt = safe_input(txt)
    if stopword:
        stopwords = sys_stopwords
    else:
        stopwords = []

    return _sentiment.predict(txt, stopword=stopwords)

def radical(txt):
    txt = safe_input(txt)
    return _radical.radical(txt)

def tag_mean(tag):
    return tag_dict.get(tag, 'undefined !')