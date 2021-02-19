# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from xmnlp.module import Module


class Radical(Module):
    __notsave__ = []
    __onlysave__ = ['dictionary']

    def __init__(self):
        self.dictionary = {}

    def train(self, fpath):
        """train model"""

        for fname in self.filelist(fpath):
            with open(fname, 'r', encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    arr = line.split(',')
                    if len(arr) != 2:
                        continue
                    self.dictionary[arr[0]] = arr[1]

    def radical(self, char):
        """get radical of given char"""

        if char in self.dictionary:
            return self.dictionary[char]
        return None
