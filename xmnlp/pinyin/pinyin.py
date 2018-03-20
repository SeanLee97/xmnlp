# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

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

from ..utils.trie import Trie

class Pinyin(object):

    def __init__(self):
        self.trie = Trie()

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

    def train(self, fpath):
        def get_file(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    if len(dirs) == 0:
                        for f in files:
                            yield os.sep.join([root, f])
            else:
                yield path

        for fname in get_file(fpath):
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if sys.version_info[0] == 2:
                        line = line.decode('utf-8')
                    words = line.split()
                    self.trie.add(words[0], words[1:])

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

    def translate(self, text):
        ret = []
        for t in self.trie.get(text):
            if isinstance(t, list) or isinstance(t, tuple):
                ret = ret + t
            else:
                ret.append(t)
        return ret