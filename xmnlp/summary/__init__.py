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

import sys
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange

import re

from .textrank import TextRank, KeywordTextRank
from .. import postag

def keyword(txt, k=10, stopword=[], allowPOS=[]):

    words = []
    for word, flag in postag.tag(txt):
        if word not in stopword:
            if len(allowPOS) > 0:
                if flag in allowPOS:
                    words.append(word)
            else:
                words.append(word)

    ktr = KeywordTextRank(words)
    return ktr.topk(k)
    


def keyphrase(txt, k=10, stopword=[]):

    def get_sents(doc):
        re_line_skip = re.compile('[\r\n]')
        re_delimiter = re.compile('[，。？！；]')
        sents = []
        for line in re_line_skip.split(doc):
            line = line.strip()
            if not line:
                continue
            for sent in re_delimiter.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                sents.append(sent)
        return sents

    sents = get_sents(txt)
    docs = []
    for sent in sents:
        words = []
        for word, flag in postag.tag(sent):
            if word not in stopword:
                words.append(word)
        docs.append(words)

    tr = TextRank(docs)
    res = []
    for idx in tr.topk(k):
        res.append(docs[idx])
    
    return res
