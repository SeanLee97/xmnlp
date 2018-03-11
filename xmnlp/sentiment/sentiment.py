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

import os

from ..utils.clf import NBayes
from ..postag import seg

class Sentiment(object):

    def __init__(self):
        self.clf = NBayes()

    def save(self, fname):
        self.clf.save(fname)

    def load(self, fname):
        self.clf.load(fname)

    def filter_stopword(self, words, stopword=[]):
        if len(stopword) == 0:
            return words
        ret = []
        for word in words:
            if word not in stopword:
                ret.append(word)
        return ret

    def load_data(self, posfname, negfname):
        def get_file(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    if len(dirs) == 0:
                        for f in files:
                            yield os.sep.join([root, f])
            else:
                yield path

        pos_docs = []
        neg_docs = []
        for fname in get_file(posfname):
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if sys.version_info[0] == 2:
                        line = line.decode('utf-8')
                    pos_docs.append(seg(line))

        for fname in get_file(negfname):
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if sys.version_info[0] == 2:
                        line = line.decode('utf-8')
                    neg_docs.append(seg(line))

        return pos_docs, neg_docs

    def train(self, posfname, negfname, stopword=[]):
        pos_docs, neg_docs = self.load_data(posfname, negfname)

        data = []
        for sent in neg_docs:
            data.append(('neg', self.filter_stopword(sent, stopword=stopword)))
        for sent in pos_docs:
            data.append(('pos', self.filter_stopword(sent, stopword=stopword)))

        self.clf.train(data)

    def predict(self, doc, stopword=[]):
        sent = seg(doc)
        ret, prob = self.clf.predict(self.filter_stopword(sent, stopword=stopword))
        if ret == 'pos':
            return prob
        return 1 - prob