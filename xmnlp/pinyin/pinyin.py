# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from xmnlp.module import Module
from xmnlp.utils.trie import Trie


class Pinyin(Module):
    __notsave__ = []
    __onlysave__ = ['trie']

    def __init__(self):
        self.trie = Trie()

    def train(self, fpath):
        """train pinyin model"""

        for fname in self.filelist(fpath):
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
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
