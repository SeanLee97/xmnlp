# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
import re
from xmnlp.utils.postag import DAG
from xmnlp.utils.postag import HMM
from xmnlp.config import regx as R

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
        self.dag.load_hmm(seg_fname=fname)

    def load_tag(self, fname):
        self.dag.load_hmm(tag_fname=fname)

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

    def seg(self, text):
        for w, _ in self.tag(text):
            yield w

    def tag(self, text):
        specials = []
        for (x, tag) in [(r[1].finditer(text), r[0]) for r in R.special_tags]:
            for xx in x:
                span = xx.span()
                specials.append((span, text[span[0]:span[1]], tag))
        if specials:
            specials = sorted(specials, key=lambda x: x[0][0])
            spans = [(text[:specials[0][0][0]], None)]
            for i in range(1, len(specials)):
                spans.append((specials[i-1][1], specials[i-1][2]))
                l, r = specials[i-1][0][1], specials[i][0][0]
                spans.append((text[l:r], None))
            spans.append((specials[-1][1], specials[-1][2]))
            spans.append((text[specials[-1][0][1]:], None))
        else:
            spans = [(text, None)]
        for t in spans:
            if not t[0]:
                continue
            if t[1] is None:
                sent = R.blank.sub("", t[0])
                if not R.skip.sub("", sent):
                    yield sent, "un"
                    continue
                for s in R.eng_digt.split(sent):
                    if not s:
                        continue
                    if R.isalpha.match(s):
                        yield s, "eng"
                    elif s.isdigit():
                        yield s, "m"
                    elif R.skip.match(s):
                        yield s, "un"
                    else:
                        for w, t in self.dag.tag(s):
                            if w:
                                yield w, t
            else:
                yield t[0], t[1]
