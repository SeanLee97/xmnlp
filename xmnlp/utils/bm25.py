# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# refrence:                                  #
#   https://github.com/isnowfy/snownlp       #
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

from math import log

class BM25(object):

    def __init__(self, docs, k1=1.5, b=0.75):
        self.N = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.N
        self.docs = docs
        self.tf = []
        self.df = {}
        self.idf = {}
        self.k1 = k1
        self.b = b

        self.build()

    def build(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                if not word in tmp:
                    tmp[word] = 0
                tmp[word] += 1
            self.tf.append(tmp)
            for k, v in tmp.items():
                if k not in self.df:
                    self.df[k] = 0
                self.df[k] += 1
        for k, v in self.df.items():
            self.idf[k] = log(self.N-v+0.5)- log(v+0.5)

    def sim(self, doc, idx):
        score = 0
        for word in doc:
            if word not in self.tf[idx]:
                continue
            d = len(self.docs[idx])
            score += (self.idf[word]*self.tf[idx][word]*(self.k1+1) / (self.tf[idx][word]+self.k1*(1-self.b+self.b*d / self.avgdl)))
        return score

    def get_sims(self, doc):
        scores = []
        for idx in range(self.N):
            score = self.sim(doc, idx)
            scores.append(score)
        return scores