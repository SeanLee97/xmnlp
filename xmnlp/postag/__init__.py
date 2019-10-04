# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
from xmnlp.config import path as C_PATH
from . import postag

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


postagger = postag.Postag()
# load DAG
postagger.load_dag(C_PATH.postag['model']['dag'])
# load seg
postagger.load_seg(C_PATH.postag['model']['seg'])
# load tag
postagger.load_tag(C_PATH.postag['model']['tag'])


def set_userdict(fpath):
    """set user dictionary"""
    postagger.userdict(fpath)


def seg(doc, hmm=True):
    """segger"""
    postagger.set_hmm(hmm)
    return list(postagger.seg(doc))


def tag(doc, hmm=True):
    """tagger"""
    postagger.set_hmm(hmm)
    word_tags = postagger.tag(doc)
    return list(word_tags)


def load(dag, seg_hmm=None, tag_hmm=None):
    """load model"""
    postagger = postag.Postag()
    postagger.load(dag, seg_hmm, tag_hmm)
