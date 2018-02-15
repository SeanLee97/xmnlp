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

import numpy as np 

from ..utils.bm25 import BM25

class KeywordTextRank(object):  
      
    def __init__(self, words, window=5, alpha=0.85, iters=300):  
        self.words = words
        self.window = window  
        self.d = alpha  
        self.vertex = set(words)
        self.edge = {}
        self.iters = iters  

        self.build_edge()
        self.build_matrix()
        self.calc_pr()
  
    def build_edge(self):  
        N = len(self.words)  
        for idx, word in enumerate(self.words):  

            if word not in self.edge.keys():  
                tmp = set()  

                l = idx - self.window + 1 # left win
                r = idx + self.window     # right win

                l = 0 if l<0 else l 
                r = N if r>=N else r

                # get window
                for i in range(l, r):  
                    if i == idx:  
                        continue  
                    tmp.add(self.words[i])  
                self.edge[word] = tmp
  
    def build_matrix(self):  
        self.matrix = np.zeros([len(self.vertex), len(self.vertex)])  
        self.word_idx = {}  #记录词的idx  
        self.idx_dict = {}  #记录节点idx对应的词  
  
        for i, v in enumerate(self.vertex):  
            self.word_idx[v] = i  
            self.idx_dict[i] = v  

        for key in self.edge.keys():  
            for w in self.edge[key]:  
                self.matrix[self.word_idx[key]][self.word_idx[w]] = 1  
                self.matrix[self.word_idx[w]][self.word_idx[key]] = 1  

        for j in range(self.matrix.shape[1]):  
            Z = 0  
            for i in range(self.matrix.shape[0]):  
                Z += self.matrix[i][j]  
            for i in range(self.matrix.shape[0]):  
                self.matrix[i][j] /= Z  
  
    def calc_pr(self):  
        self.PR = np.ones([len(self.vertex), 1])  
        for i in range(self.iters):  
            self.PR = (1 - self.d) + self.d * np.dot(self.matrix, self.PR)  
  
    #输出词和相应的权重  
    def topk(self, k):  
        word_pr = {}  
        for i in range(len(self.PR)):  
            word_pr[self.idx_dict[i]] = self.PR[i][0]  
        res = sorted(word_pr.items(), key = lambda x : x[1], reverse=True)  
        return res[:k]  

class TextRank(object):

    def __init__(self, docs, alpha=0.85, min_diff=1e-2, iters=500):
        self.docs = docs
        self.bm25 = BM25(docs)
        self.N = len(docs)
        self.d = alpha
        self.weight = []
        self.weight_sum = []
        self.vertex = []
        self.iters = iters
        self.min_diff = min_diff

        self.build()
        self.calc_pr()

    def build(self):
        for idx, doc in enumerate(self.docs):
            scores = self.bm25.get_sims(doc)
            self.weight.append(scores)
            self.weight_sum.append(sum(scores)-scores[idx])
            self.vertex.append(1.0)

    def calc_pr(self):    
        for _ in range(self.iters):
            m = []
            max_diff = 0
            for i in range(self.N):
                m.append(1 - self.d)
                for j in range(self.N):
                    if j == i or self.weight_sum[j] == 0:
                        continue
                    m[-1] += (self.d * self.weight[j][i] / self.weight_sum[j]*self.vertex[j])
                if abs(m[-1] - self.vertex[i]) > max_diff:
                    max_diff = abs(m[-1] - self.vertex[i])
            self.vertex = m
            if max_diff <= self.min_diff:
                break

    def topk(self, k):
        res = list(enumerate(self.vertex))
        res = sorted(res, key=lambda x: x, reverse=True)
        return list(map(lambda x: x[0], res))[:k]