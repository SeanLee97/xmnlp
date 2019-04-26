# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import re
import sys
from .textrank import TextRank, KeywordTextRank
from .. import postag

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange


def keyword(text, k=10, stopword=None, allowPOS=None):
    """extract keyword from text"""

    if stopword is None:
        stopword = []
    if allowPOS is None:
        allowPOS = []

    words = []
    for word, tag in postag.tag(text):
        if word not in stopword:
            if allowPOS is not None:
                if tag in allowPOS:
                    words.append(word)
            else:
                words.append(word)

    words = KeywordTextRank(words)
    return words.topk(k)


def keyphrase(text, k=10, stopword=None):
    """extract keyphrase from text"""

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

    if stopword is None:
        stopword = []
    sents = get_sents(text)
    docs = []
    for sent in sents:
        words = []
        for word, _ in postag.tag(sent):
            if word not in stopword:
                words.append(word)
        docs.append(words)

    tr = TextRank(docs)
    res = []
    for idx in tr.topk(k):
        res.append(docs[idx])
    
    return res
