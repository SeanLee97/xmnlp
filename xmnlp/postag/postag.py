# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
import re
from ..utils.postag import DAG
from ..utils.postag import HMM
from ..config import regx as R

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange


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

    def train_hmm(self, srcfname, outfname):
        """train hmm"""
        hmm = HMM()
        hmm.train(srcfname)
        hmm.save(outfname)

    def set_hmm(self, hmm=True):
        self.dag.set_hmm(hmm)

    def re_decode(self, parts, arr):
        if parts[0]:
            for x in parts[0]:
                yield x, 'un'
        for x, y in zip(parts[1:], arr):
            if y.isalpha():
                yield y, 'eng'
            else:
                yield y, 'm'
            for xx in x:
                yield xx, 'un'

    def seg(self, sent):
        for s in R.zh.split(sent):
            s = s.strip()
            if not s:
                continue
            if R.zh.match(s):
                for w in list(self.dag.seg(s)):
                    if w.strip():
                        yield w
            else:
                tmp = R.skip.split(s)
                for x in tmp:
                    if R.skip.match(x):
                        if x.strip():
                            yield x
                    else:
                        x = x.replace(' ','')
                        endigts = R.endigt.findall(x)
                        parts = re.split(r'[0-9]+\.?[0-9]+|[0-9]+|[a-zA-Z]+', x)
                        if endigts:
                            for w, _ in self.re_decode(parts, endigts):
                                if w.strip():
                                    yield w
                        else:
                            for xx in x:
                                if xx.strip():
                                    yield xx

    def tag(self, sent):
        for s in R.zh.split(sent):
            s = s.strip()
            s = R.skip.sub('', s)

            if not s:
                continue
            if R.zh.match(s):
                for w, t in self.dag.tag(s):
                    if w.strip():
                        yield w, t
            else:
                tmp = R.skip.split(s)
                for x in tmp:
                    if R.skip.match(x):
                        if x.strip():
                            yield x
                    else:
                        x = x.replace(' ', '')
                        endigts = R.endigt.findall(x)
                        parts = re.split(r'[0-9]+\.?[0-9]+|[0-9]+|[a-zA-Z]+', x)
                        if endigts:
                            for w, t in self.re_decode(parts, endigts):
                                if w.strip():
                                    yield w, t
                        else:
                            for xx in x:
                                if xx.strip():
                                    yield xx, 'un'
