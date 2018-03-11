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
import os
from math import log

from collections import defaultdict

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'

"""bigram / trigram hmm, viterbi decode"""

class HMM(object):
    def __init__(self, N=2, bems=True, *args):

        self.N = 2
        if N == 3:
            self.N == N 
            
        self.bems = bems
        self.line_total = -1

        # n-gram counter
        self.uni = defaultdict(int)
        self.bi = defaultdict(int)
        self.tri = defaultdict(int)

        # emit - <(word,tag), times>
        self.emit = defaultdict(int)
        self.state = set()
        self.words = set()

        # emit probs
        self._emit_prob = defaultdict(int)
        # trans probs
        self._trans_prob = defaultdict(int)

        self._start_prob = {}

    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                d[k] = list(v)
            elif hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v

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

        for k, v in d.items():
            if 'bems' == k:
                continue

            if isinstance(self.__dict__[k], set):
                self.__dict__[k] = set(v)
            elif hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def load_data(self, fpath):
        def get_file(path):
            if os.path.isdir(path):
                 for root, dirs, files in os.walk(path):
                     if len(dirs) == 0:
                         for f in files:
                             yield os.sep.join([root, f])
            else:
                yield path

        datas = []
        for fname in get_file(fpath):
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    self.line_total += 1
                    datas.append(map(lambda x: x.split('/'), line.split()))

        return datas

    def train(self, fname):

        '''
        def _filter(freq):
            new = defaultdict(int)
            for (word,tag) in self.emit:
                new[(word,tag)] = self.emit[(word,tag)]
                if self.emit[(word,tag)] < freq:
                    new[(UNK,tag)] += self.emit[(word,tag)]
            self.emit = new
        '''

        datas = self.load_data(fname)            

        for data in datas:
            # init curr
            curr = [BOS] * 2
            self.bi[tuple(curr)] += 1
            self.uni[BOS] += 2

            for word, tag in data:
                curr.append(tag)

                self.state.add(tag)
                self.emit[(word, tag)] += 1
                self.uni[tag] += 1
                self.bi[tuple(curr[1:])] += 1
                self.tri[tuple(curr)] += 1

                curr.pop(0)

        self.words = set([key[0] for key in self.emit.keys()])
        
        for (word,tag) in self.emit:
            self._emit_prob[(word,tag)] = self.calc_e(word,tag)

        if self.N == 3:
            for gram in list(self.tri):
                self._trans_prob[gram] = self.calc_trans(gram)
        else:
            for gram in list(self.bi):
                self._trans_prob[gram] = self.calc_trans(gram)

        for s in self.state:
            t = self.uni.get(s, 0)
            t = log(t) if self.bems else t
            self._start_prob[s] = t * 1.0 / self.line_total


    def calc_e(self, word, tag):
        x = self.emit[(word,tag)]
        y = self.uni[tag]
        return float(x + 1) / (y + len(self.words)*len(self.uni))

    def calc_trans(self, gram):
        if self.N == 3:
            x = self.tri[tuple(gram)]
            y = self.bi[tuple(gram[:2])]
        else:
            x = self.bi[tuple(gram)]
            y = self.uni[gram[0]]
        return (x + 1.0) / (y + len(self.state))

    def viterbi(self, sent):
        if self.bems:
            self._start_prob['m'] = 0.0
            self._start_prob['e'] = 0.0

        def get_states(k):
            if k <= 0:
                return set([BOS])
            else:
                return self.state

        def get_word(sent,k):
            if k < 0:
                return BOS
            else:
                return sent[k]

        def smooth(x):
            return x + 1e-12

        V = {}
        path = {}

        if self.N == 3:
            # init
            V[0, BOS, BOS] = 1
            path[BOS, BOS] = []
            # run
            for k in range(1, len(sent)+1):
                temp_path = {}
                word = get_word(sent,k-1)
                if word not in self.words:
                    word = UNK
                for u in get_states(k-1):
                    for v in get_states(k):
                        V[k,u,v], prev_w = max([(V[k-1,w,u] * smooth(self._trans_prob[(w,u,v)] * self._emit_prob[(word,v)]), w) for w in get_states(k-2) if V[k-1, w] > 0])
                        temp_path[u,v] = path[prev_w, u] + [v]
                path = temp_path
            # last step
            prob,umax,vmax = max([(V[len(sent),u,v] * self._trans_prob[(u,v,BOS)],u, v) for u in self.state for v in self.state])
            return (prob, path[umax,vmax])
        else:
            V = [{}]
            path = {}
            for s in self.state:
                V[0][s] = self._start_prob[s] * smooth(self._emit_prob[(sent[0], s)])
                path[s] = [s]
            N = len(sent)
            # run
            for k in range(1, N):
                V.append({})
                temp_path = {}
                word = get_word(sent, k)
                if word not in self.words:
                    word = UNK
                for u in get_states(k):
                    V[k][u], state = max([(V[k-1][w] * smooth(self._trans_prob[(w, u)] * self._emit_prob[(word, u)]), w) for w in get_states(k) if V[k-1][w] > 0])
                    temp_path[u] = path[state] + [u]
                path = temp_path
                #print(path)
            prob, state = max([(V[N-1][u], u) for u in self.state])
            return (prob, path[state])

    def tag(self, sent):
        prob, tags = self.viterbi(sent)
        return zip(list(sent), tags)
