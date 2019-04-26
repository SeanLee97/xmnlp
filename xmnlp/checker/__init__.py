# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
from ..config import path as C_PATH
from ..config import regx as R
from ..postag import seg
from .checker import Checker

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

# checker model
model = None

def loader():
    """load model"""

    global model
    if model is None:
        model = Checker()

def set_userdict(fpath):
    """set user dict"""

    loader()
    model.userdict(fpath)

def check(doc, level=0):
    """check doc

    Args:
      level:
        - 0: word
        - 1: doc
    """

    loader()
    if isinstance(doc, (str, unicode)):
        if level == 0:
            return model.best_match(doc)
        return model.doc_checker(doc)
    raise ValueError('Error [Chekcer]: invalid input type, str is required!')
