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

import os
from ..utils.trie import Trie

class PinYin(object):

    def __init__(self, fname):
        self.trie = Trie()

        self.load(fname)

    def load(self, fpath):
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


    def translate(self, text):
        ret = []
        for t in self.trie.get(text):
            if isinstance(t, list) or isinstance(t, tuple):
                ret = ret + t
            else:
                ret.append(t)
        return ret