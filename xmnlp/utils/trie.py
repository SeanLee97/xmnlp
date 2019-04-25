# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import unicode_literals
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
                if lst:
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
                    if cands:
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
