# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import unicode_literals
import sys
from .pinyin import Pinyin
from ..config import path as C_PATH, regx as R

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


PY = None
def loader():
    """load model"""

    global PY
    if PY is None:
        PY = Pinyin()
        model_path = C_PATH.pinyin['model']['pinyin']
        PY.load(model_path)


def translate(sent):
    """translate chinese to pinyin"""

    loader()
    ret = []
    for s in R.zh.split(sent):
        s = s.strip()
        if not s:
            continue
        if R.zh.match(s):
            ret += PY.translate(s)
        else:
            for word in s.split():
                word = word.strip()
                if word:
                    ret.append(word)
    return ret
