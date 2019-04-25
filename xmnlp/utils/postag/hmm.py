# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
import io
from math import log
from collections import defaultdict
from xmnlp.module import Module
from xmnlp.utils import safe_input

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'


class HMM(Module):
    __notsave__ = []
    __onlysave__ = []

    def __init__(self, N=2, bems=True):
        self.N = 2
        if N == 3:
            self.N = N
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

    def load_data(self, fpath):
        datas = []
        for fname in self.filelist(fpath):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = safe_input(line)
                    if not line:
                        continue
                    self.line_total += 1
                    datas.append(list(map(lambda x: x.split('/'), line.split())))
        return datas

    def train(self, fname):
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
        for (word, tag) in self.emit:
            self._emit_prob[(word, tag)] = self.calc_e(word, tag)
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
        x = self.emit[(word, tag)]
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

        def get_word(sent, k):
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
                word = get_word(sent, k-1)
                if word not in self.words:
                    word = UNK
                for u in get_states(k-1):
                    for v in get_states(k):
                        V[k, u, v], prev_w = max([(V[k-1, w, u] * smooth(self._trans_prob[(w, u, v)] * self._emit_prob[(word, v)]), w) for w in get_states(k-2) if V[k-1, w] > 0])
                        temp_path[u, v] = path[prev_w, u] + [v]
                path = temp_path
            # last step
            prob, umax, vmax = max([(V[len(sent), u, v] * self._trans_prob[(u, v, BOS)], u, v) for u in self.state for v in self.state])
            return (prob, path[umax, vmax])
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
        _, tags = self.viterbi(sent)
        return zip(list(sent), tags)
