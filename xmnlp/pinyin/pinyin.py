# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import unicode_literals
import sys
import io
from ..module import Module
from ..utils.trie import Trie
from ..utils import safe_input

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange

class Pinyin(Module):
    __notsave__ = []
    __onlysave__ = ['trie']

    def __init__(self):
        self.trie = Trie()

    def train(self, fpath):
        """train pinyin model"""

        for fname in self.filelist(fpath):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = safe_input(line)
                    words = line.split()
                    self.trie.add(words[0], words[1:])

    def translate(self, text):
        """translate chinese charactor to pinyin """
        ret = []
        for t in self.trie.get(text):
            if isinstance(t, (list, tuple)):
                ret = ret + t
            else:
                ret.append(t)
        return ret
