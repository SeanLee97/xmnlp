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

from ..utils.postag import DAG
from ..config import regx as R

class Postag(object):

    def __init__(self):
        self.dag = DAG()

    def userdict(self, fpath):
        self.dag.userdict(fpath)
    
    def load(self, fname, seg_hmm=None, tag_hmm=None):
        self.dag.load(fname)
        self.dag.load_hmm(seg_hmm, tag_hmm)
        
    def load_dag(self, fname):
        self.dag.load(fname)

    def load_seg(self, fname):
        self.dag.load_hmm(segfname=fname)

    def load_tag(self, fname):
        self.dag.load_hmm(tagfname=fname)

    def train(self, srcfname, outfname):
        self.dag.train(srcfname)
        self.dag.save(outfname)

    # train hmm
    def train_hmm(self, srcfname, outfname):
        from ..utils.postag import HMM
        hmm = HMM()
        hmm.train(srcfname)
        hmm.save(outfname)

    def set_hmm(self, hmm=True):
        self.dag.set_hmm(hmm)

    def re_decode(self, x, arr, tag=None):
        for d in arr:
            x = x.replace(d, ' ')
        
        if len(x.strip()) == 0:
            if tag != None:
                yield arr[0], tag
            else:
                yield arr[0]
        else:
            for idx, xx in enumerate(x.split()):
                if tag != None:
                    yield xx, 'un'
                    if idx < len(arr):
                        yield arr[idx], tag
                else:
                    yield xx
                    if idx in arr:
                        yield arr[idx]

    def seg(self, sent):
        for s in R.zh.split(sent):
            s = s.strip()
            if not s:
                continue
            if R.zh.match(s):
                for w in list(self.dag.seg(s)):
                    yield w
            else:
                tmp = R.skip.split(s)
                for x in tmp:
                    if R.skip.match(x):
                        yield x
                    else:
                        digts = R.digt.findall(x)
                        engs = R.eng.findall(x)
                        x = x.replace(' ','')
                        if len(digts) > 0:
                            for w, t in self.re_decode(x, digts, 'm'):
                                yield w
                        elif len(engs) > 0:
                            for w, t in self.re_decode(x, engs, 'eng'):
                                yield w
                        else:
                            for xx in x:
                                yield xx
    def tag(self, sent):
        for s in R.zh.split(sent):
            s = s.strip()
            s = R.skip.sub("", s)

            if not s:
                continue
            if R.zh.match(s):
                for w,t in self.dag.tag(s):
                    yield w, t

            else:
                tmp = R.skip.split(s)
                for x in tmp:
                    if R.skip.match(x):
                        yield x
                    else:
                        digts = R.digt.findall(x)
                        engs = R.eng.findall(x)
                        x = x.replace(' ','')
                        if len(digts) > 0:
                            for w, t in self.re_decode(x, digts, 'm'):
                                yield w, t
                        elif len(engs) > 0:
                            for w, t in self.re_decode(x, engs, 'eng'):
                                yield w, t
                        else:
                            for xx in x:
                                yield xx, 'un'