# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
import io
from ..module import Module

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange


class Radical(Module):
    __notsave__ = []
    __onlysave__ = ['dictionary']

    def __init__(self):
        self.dictionary = {}

    def train(self, fpath):
        """train model"""

        for fname in self.filelist(fpath):
            with io.open(fname, 'r', encoding="utf-8") as f:
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
