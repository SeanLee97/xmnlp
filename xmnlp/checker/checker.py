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

from ..pinyin import translate

import os
import gzip

from math import log
from collections import defaultdict

BOS = u'<BOS>'
EOS = u'<EOS>'

class Checker(object):

    def __init__(self):

        # n-gram counter
        self.uni = defaultdict(int)
        self.bi = defaultdict(int)

        self.pinyins = {}
        self.chs = set()
        self.words = defaultdict(int)

        self.threhold = 1*1e-7

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
                    if sys.version_info[0] == 2:
                        line = line.decode('utf-8')

                    if len(line) == 0:
                        continue
                    datas.append(line.split())

        return datas

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
            if isinstance(self.__dict__[k], set):
                self.__dict__[k] = set(v)
            elif hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def train(self, fname):
        datas = self.load_data(fname)
        for words in datas:
            for word in words:
                if word not in self.words:
                    self.words[word] += 1
                else:
                    self.words[word] += 1

            data = ''.join(words)
            # init curr
            curr = [BOS]
            self.uni[BOS] += 1

            for word in data:
                curr.append(word)
                self.uni[word] += 1
                self.bi[tuple(curr)] += 1

                curr.pop(0)

        for c in self.uni.keys():
            if c == BOS:
                continue
            self.chs.add(c)
            py = ''.join(translate(c))
            if py not in self.pinyins:
                self.pinyins[py] = [c]
            else:
                self.pinyins[py].append(c)

    def calc_proba(self, gram):
        x = self.bi[tuple(gram)]
        y = self.uni[gram[0]]
        return float((x + 1)) / (y + len(self.uni.keys())**2)

    def get_ngram(self, sent):
        ngram = []

        curr = [BOS]
        for word in sent:
            curr.append(word)
            ngram.append( tuple(curr) )
            curr.pop(0)

        return ngram        

    def levenshtein(self, phrase):
        splits     = [(phrase[:i], phrase[i:])  for i in range(len(phrase) + 1)]
        deletes    = [L + R[1:]                 for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:]   for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]             for L, R in splits if R for c in self.chs]
        inserts    = [L + c + R                 for L, R in splits for c in self.chs]
        
        return set(deletes + transposes + replaces + inserts)


    def ngram_proba(self, words):
        #score = 1.0
        probs = []
        ngram = self.get_ngram(words)
        for g in ngram:
            p = self.calc_proba(g)
            #score *= p
            probs.append((g, p))
        return probs

    def correct(self, words, window=4):
        if len(words) < 1:
            return ''.join(words)

        res = ''
        pos = 0
        probs = {}
        while pos < len(words):
            sent = ''.join(words[pos: pos + window])
            if len(sent) == 1:
                res += sent
                pos += window
                continue

            probs = self.ngram_proba(sent)
            wrongs = []
            idx = 0
            while idx < len(probs):
                if idx+1 == len(probs):
                    break
                if probs[idx][0][0] not in [BOS, EOS] and probs[idx+1][0][0] not in [BOS, EOS] \
                and probs[idx][1] < self.threhold and probs[idx+1][1] < self.threhold:
                    wrongs.append((probs[idx][0], probs[idx+1][0]))
                
                idx += 1

            correct = {}
            for pair in wrongs:
                first = pair[0]
                second = pair[1]
                if first[1] == second[0]:
                    candidates = []
                    wrong = first[1]
                    wrong_py = ''.join(translate(wrong))
                    if wrong_py not in self.pinyins:
                        continue

                    fch  = correct[first[0]]  if first[0] in correct.keys() else first[0]
                    for ch in self.pinyins[wrong_py]:
                        p = self.calc_proba((fch, ch)) + self.calc_proba((ch, second[1]))
                        candidates.append((ch, p))
                    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    correct[wrong] = candidates[0][0]

            for w in sent:
                res += correct.get(w, w)

            pos += window

        # process head and tail
        ps = []
        if probs[1][1] < self.threhold:
            ps.append((''.join(probs[1][0]), probs[1]))
        if probs[-1][1] < self.threhold:
            ps.append((''.join(probs[-1][0]), probs[-1]))
        for word, prob in ps:
            if prob[1] < self.threhold:
                py = ''.join(translate(word))
                edits = list(filter(lambda x: len(x) == len(word) and ''.join(translate(x)) == py, self.levenshtein(word)))
                if len(edits) > 0:
                    candidates = []
                    for ch in edits:
                        if ch in self.words:
                            p = self.calc_proba((ch[0], ch[1]))
                            candidates.append((ch, p))
                        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                    if len(candidates) > 0 and candidates[0][1] > prob[1]:
                        res = res.replace(word, candidates[0][0])
        return res
