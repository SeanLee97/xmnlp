# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import sys
from ..config import path as C_PATH
from . import postag

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

postagger = None
segger = None
tagger = None

def dag_loader():
    """load dag model"""

    global postagger
    if postagger is None:
        postagger = postag.Postag()
        model_path = C_PATH.postag['model']['dag']
        postagger.load_dag(model_path)


def seg_loader():
    """load seg model"""

    dag_loader()
    global segger
    global postagger
    if segger is None:
        model_path = C_PATH.postag['model']['seg']
        postagger.load_seg(model_path)
        segger = postagger


def tag_loader():
    """load tag model"""

    # tag 依赖于segger
    seg_loader()
    global tagger
    global postagger
    if tagger is None:
        model_path = C_PATH.postag['model']['tag']
        postagger.load_tag(model_path)
        tagger = postagger


def set_userdict(fpath):
    """set user dictionary"""

    # 依赖于segger
    seg_loader()
    global postagger
    postagger.userdict(fpath)


def seg(doc, hmm=True):
    """segger"""

    seg_loader()
    segger.set_hmm(hmm)
    words = list(segger.seg(doc))
    return words


def tag(doc, hmm=True):
    """tagger"""

    tag_loader()
    tagger.set_hmm(hmm)
    word_tags = tagger.tag(doc)
    return list(word_tags)

def load(dag, seg_hmm=None, tag_hmm=None):
    """load model"""

    global postagger, segger, tagger
    postagger = postag.Postag()
    postagger.load(dag, seg_hmm, tag_hmm)
    # update
    segger = postagger
    tagger = postagger
