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
    import cPickle as pickle
else:
    import pickle

import gzip

from math import log, exp
from collections import defaultdict

class NBayes(object):

    def __init__(self):
        self.corpus = {}
        self.counter = {}
        self.total = 0

    def save(self, fname, iszip=True):
        d = {}
        d['counter'] = self.counter
        d['corpus'] = self.corpus
        d['total'] = self.total

        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            pickle.dump(d, open(fname, 'wb'), True)
        else:
            f = gzip.open(fname, 'wb')
            f.write(pickle.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = pickle.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = pickle.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = pickle.loads(f.read())
            f.close()
        self.counter = d['counter']
        self.corpus = d['corpus']
        self.total = d['total']

    def train(self, data):
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

    def predict(self, sent):
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
