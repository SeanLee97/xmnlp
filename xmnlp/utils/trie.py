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

class Trie(object):

    def __init__(self):
        self.root = {}

    def add(self, key, val):
        curr = self.root
        for k in key:
            if not k in curr:
                curr[k] = {}
            curr = curr[k]
        curr['val'] = val

    def find(self, sent, start=0):
        curr = self.root
        ret = None
        for pos in range(start, len(sent)):
            if sent[pos] in curr:
                curr = curr[sent[pos]]
            else:
                return ret
            if 'val' in curr:
                ret = (sent[start:pos+1], curr['val'])
            else:
                lst = list(curr)
                if len(lst) > 0:
                    cands = {}
                    for c in lst:
                        if 'val' not in curr[c]:
                            continue
                        py = curr[c]['val'][0]
                        if py not in cands:
                            cands[py] = 1
                        else:
                            cands[py] += 1
                    cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)
                    if len(cands) > 0:
                        ret = (sent[start:pos+1], cands[0][0])   
        return ret

    def get(self, sent):
        ret = []
        pos = 0
        while pos < len(sent):
            curr = self.root
            if sent[pos] in curr:
                tmp = self.find(sent, pos)
                if tmp:
                    ret.append(tmp[1])
                    pos += len(tmp[0])
                    continue
            pos += 1
        return ret