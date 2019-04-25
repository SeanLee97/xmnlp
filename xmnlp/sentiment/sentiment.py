# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import io
import os
import sys
from math import log, exp
from collections import defaultdict
from ..module import Module
from ..utils import safe_input
from ..postag import seg

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


class NBayes(Module):
    __notsave__ = []
    __onlysave__ = ['counter', 'corpus', 'total']

    def __init__(self):
        self.corpus = {}
        self.counter = {}
        self.total = 0

    def process_data(self, data):
        """process train data"""

        for d in data:
            label = d[0]
            doc = d[1]
            if label not in self.corpus:
                self.corpus[label] = defaultdict(int)
                self.counter[label] = 0
            for word in doc:
                self.counter[label] += 1
                self.corpus[label][word] += 1
        self.total = sum(self.counter.values())

    def calc_score(self, sent):
        """calculate sentiment score"""

        tmp = {}
        for k in self.corpus:
            tmp[k] = log(self.counter[k]) - log(self.total)
            for word in sent:
                x = float(self.corpus[k].get(word, 1)) / self.counter[k]
                tmp[k] += log(x)

        ret, prob = 0, 0
        for k in self.corpus:
            curr = 0
            try:
                for kk in self.corpus:
                    curr += exp(tmp[kk] - tmp[k])
                curr = 1.0 / curr
            except OverflowError:
                curr = 0.0
            if curr > prob:
                ret, prob = k, curr
        return (ret, prob)

class Sentiment(NBayes):
    def filter_stopword(self, words, stopword=None):
        """filter stopwords"""

        if stopword is None:
            return words
        ret = [word for word in words if word not in stopword]
        for word in words:
            if word not in stopword:
                ret.append(word)
        return ret

    def load_data(self, posfname, negfname):
        """load dataset from file"""

        def get_file(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    if not dirs:
                        for f in files:
                            yield os.sep.join([root, f])
            else:
                yield path

        pos_docs, neg_docs = [], []
        for fname in get_file(posfname):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = safe_input(line)
                    pos_docs.append(seg(line))
        for fname in get_file(negfname):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = safe_input(line)
                    neg_docs.append(seg(line))

        return pos_docs, neg_docs

    def train(self, posfname, negfname, stopword=None):
        """train sentiment model"""

        pos_docs, neg_docs = self.load_data(posfname, negfname)
        data = []
        for sent in neg_docs:
            data.append(('neg', self.filter_stopword(sent, stopword=stopword)))
        for sent in pos_docs:
            data.append(('pos', self.filter_stopword(sent, stopword=stopword)))

        self.process_data(data)

    def predict(self, doc, stopword=None):
        """predict sentiment score"""

        sent = seg(doc)
        ret, prob = self.calc_score(self.filter_stopword(sent, stopword=stopword))
        if ret == 'pos':
            return prob
        return 1 - prob
