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


from ..config import path as C_PATH
from ..config import regx as R
from ..postag import seg
from .checker import Checker

chkr = None
def loader():
    global chkr
    if chkr == None:
        try:
            chkr = Checker()
            model_path = C_PATH.checker['model']['checker']
            chkr.load(model_path)
        except:
            pass

def check(doc):
    loader()
    if isinstance(doc, str) or isinstance(doc, unicode):
        zhs = []
        pos = []
        for idx, s in enumerate(doc):
            s = s.strip()
            if not s:
                continue
            if R.zh.match(s):
                zhs.append(s)
            else:
                pos.append((idx, s))

        res = ''
        if len(zhs) > 0:
            res = chkr.correct(seg(''.join(zhs)))
            for pair in pos:
                res = res[:pair[0]] + pair[1] + res[pair[0]:]
        return res
    else:
        raise ValueError('Error [Chekcer]: invalid input type, str is required!')

def load(fname):
    global chkr 
    chkr = Checker()
    chkr.load(fname)