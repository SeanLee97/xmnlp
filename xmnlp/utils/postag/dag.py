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

import os
import gzip 
 

from math import log
from .hmm import HMM

class DAG(object):

    def __init__(self, *args):
        self.hmm = True

        self.default_dict = {}

    def set_hmm(self, hmm=True):
        self.hmm = hmm

    def train(self, fname):
        counter, total, word_tag = self.load_dict(fname)
        self.default_dict = {
            'counter': counter,
            'total': total,
            'word_tag': word_tag
        }

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

        self.dict = self.default_dict

    def load_hmm(self, segfname=None, tagfname=None):
        if segfname == None and tagfname == None:
            self.hmm = False
            return

        if segfname != None:
            try:
                self.hmm_segger = HMM(bems=True)
                self.hmm_segger.load(segfname)
            except Exception as e:
                print('Error: load seg hmm model failed, ', e)
                self.hmm = False

        if tagfname != None:
            try:
                self.hmm_tagger = HMM(bems=False)
                self.hmm_tagger.load(tagfname)
            except Exception as e:
                print('Error: load tag hmm model failed, ', e)
                self.hmm = False

    def get_freq(self, word):
        total = float(self.default_dict['total'])
        freq = 1
        for w in self.seg(word):
            freq *= self.default_dict['counter'].get(w, 1) / total
        freq = max(int(freq * total), self.default_dict['counter'].get(word, 1))
        return freq

    def userdict(self, fpath):
        counter, total, word_tag = self.load_dict(fpath)
        self.dict = {
            'counter': dict(self.dict['counter'], **counter),
            'total': self.default_dict['total'] + total,
            'word_tag': dict(self.dict['word_tag'], **word_tag)
        }

    def load_dict(self, fpath):
        def get_file(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    if len(dirs) == 0:
                        for f in files:
                            yield os.sep.join([root, f])
            else:
                yield path

        counter = {}
        total = 0
        word_tag = {}
        
        for fname in get_file(fpath):
            with open(fname, 'r') as f:
                for idx, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        if sys.version_info[0] == 2:
                            line = line.decode('utf-8')

                        arr = line.split()
                        if len(arr) >= 3:
                            word, freq, tag = arr[:3]
                        elif len(arr) == 2:
                            word, x = arr[:2]
                            if x.isdigit():
                                freq = x
                                tag = 'un'
                            else:
                                freq = self.get_freq(word)
                                tag = x

                        elif len(arr) == 1:
                            word = arr[0]
                            freq = self.get_freq(word)
                            tag = 'un'
                        else:
                            continue

                        freq = int(freq)
                        counter[word] = freq
                        total += freq
                        word_tag[word] = tag

                        # trie
                        for ch in range(len(word)):
                            wfrag = word[:ch + 1]
                            if wfrag not in counter:
                                counter[wfrag] = 0
                    except ValueError:
                        raise ValueError('invalid dictionary entry in %s at Line %s: %s' % (fname, idx, line))
        return counter, total, word_tag

    def get_dag(self, sent):
        dag = {}
        N = len(sent)

        for k in range(N):
            tmp = []
            i = k
            s = sent[k]
            while i < N and s in self.dict['counter']:
                if self.dict['counter'][s]:
                    tmp.append(i)
                i += 1
                s = sent[k: i+1]
            if not tmp:
                tmp.append(k)
            dag[k] = tmp
        return dag

    def get_route(self, sent, dag, r=None, reverse=True):
        route = {}
        N = len(sent)
        log_total = log(self.dict['total'])

        if reverse:
            route[N] = (1, 0)
            if r != None:
                r[N] = (1, 0)
            for idx in range(N - 1, -1, -1):
                if r != None:
                    route[idx] = max(((log(self.dict['counter'].get(sent[idx: x+1]) or 1) - log_total + route[x+1][0]) + r[x+1][0], x) for x in dag[idx])
                else:
                    route[idx] = max((log(self.dict['counter'].get(sent[idx: x+1]) or 1) - log_total + route[x+1][0], x) for x in dag[idx])
        else:
            for idx in range(N):
                route[idx] = (log(self.dict['counter'].get(sent[idx]) or 1) - log_total, 0)

            for idx in range(N):
                if r != None:
                    route[idx] = max(((log(self.dict['counter'].get(sent[idx: x-1]) or 1) - log_total + route[x][0]) + r[x][0], x) for x in dag[idx])
                else:
                    route[idx] = max((log(self.dict['counter'].get(sent[idx: x-1]) or 1) - log_total + route[x][0], x) for x in dag[idx])
        return route

    def tag(self, sent):
        s = self.seg(sent)
        lst = list(s)
        ret = self.hmm_tagger.tag(lst)
        for w, f in ret:
            yield w, self.dict['word_tag'].get(w, f)

    def seg(self, sent):
        return self.decode(sent)

    def decode(self, sent):
        dag = self.get_dag(sent)
        r = self.get_route(sent, dag, reverse=False)
        route = self.get_route(sent, dag, r=r, reverse=True)

        x = 0
        buf = ''
        N = len(sent)
        while x < N:
            y = route[x][1] + 1
            l_word = sent[x:y]
            if y - x == 1:
                buf += l_word
            else:
                if buf:
                    if len(buf) == 1:
                        yield buf
                        buf = ''
                    else:
                        if not self.dict['counter'].get(buf):
                            if self.hmm:
                                for elem in self.hmm_seg(buf):
                                    yield elem
                            else:
                                for elem in buf:
                                    yield elem
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''
                yield l_word
            x = y
        if buf:
            if len(buf) == 1:
                yield buf
            else:
                if not self.dict['counter'].get(buf):
                    if self.hmm:
                        for elem in self.hmm_seg(buf):
                            yield elem
                    else:
                        for elem in buf:
                            yield elem
                else:
                    for elem in buf:
                        yield elem

    def hmm_seg(self, sent):
        ret = self.hmm_segger.tag(sent)
        tmp = ''
        for i in ret:
            if i[1] == 'e':
                yield tmp+i[0]
                tmp = ''
            elif i[1] == 'b' or i[1] == 's':
                if tmp:
                    yield tmp
                tmp = i[0]
            else:
                tmp += i[0]
        if tmp:
            yield tmp
