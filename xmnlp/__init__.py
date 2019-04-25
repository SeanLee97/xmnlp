# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals
import io
import os
import sys
from functools import partial
from multiprocessing import cpu_count, Pool
from .config import path as C_PATH
from .config.tag import tag_dict
from . import postag as _postag
from . import summary as _summary
from . import checker as _checker
from . import sentiment as _sentiment
from . import pinyin as _pinyin
from . import radical as _radical
from .utils import safe_input, filelist

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

__author__ = 'sean lee'
__version__ = '0.2.0'


def load_stopword(fpath):
    """load stopwords from file """

    stopwords = set()
    for fname in filelist(fpath):
        with io.open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = safe_input(line)
                stopwords.add(line)
    return stopwords


process_pool = None
ALLOW_POS = ['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
SYS_STOPWORDS = load_stopword(C_PATH.stopword['corpus']['stopword'])


def set_process_pool(n_jobs=-1):
    """setup process pool
    Args:
      - n_jobs: pool size
    """

    global process_pool
    if n_jobs == -1:
        n_jobs = cpu_count()
    process_pool = Pool(n_jobs)


def set_stopword(fpath):
    """set stopwords from file"""
    global SYS_STOPWORDS
    words = load_stopword(fpath)
    SYS_STOPWORDS = set(words) | SYS_STOPWORDS


def set_userdict(fpath):
    """set user dict"""

    _postag.set_userdict(fpath)


def seg(text, hmm=True):
    """word segmentation"""

    text = safe_input(text)
    return _postag.seg(text, hmm)

def seg_parallel(texts, hmm=False, n_jobs=-1):
    """seg texts parallel
    Args:
      - hmm: bool, whether to use hmm to detect new words
      - n_jobs: pool size of multi-process
    Return:
      generator
    """

    if not isinstance(texts, (list, tuple)):
        raise ValueError("You should pass a list or tuple texts")
    if process_pool is None:
        set_process_pool(n_jobs)
    seg_func = partial(seg, hmm=hmm)
    for words in process_pool.map(seg_func, texts):
        yield words


def tag(text, hmm=True):
    """word tagging"""

    text = safe_input(text)
    return _postag.tag(text, hmm)


def tag_parallel(texts, hmm=False, n_jobs=-1):
    """tag texts parallel
    Args:
      - hmm: bool, whether to use hmm to detect new words
      - n_jobs: pool size of multi-process
    Return:
      generator
    """
    if not isinstance(texts, (list, tuple)):
        raise ValueError("You should pass a list or tuple texts")
    if process_pool is None:
        set_process_pool(n_jobs)
    tag_func = partial(tag, hmm=hmm)
    for ret in process_pool.map(tag_func, texts):
        yield ret


def keyword(text, k=10, stopword=True, allowPOS=None):
    """extract keyword from text"""

    if allowPOS is None:
        allowPOS = ALLOW_POS
    text = safe_input(text)
    stopwords = SYS_STOPWORDS if stopword else []

    return _summary.keyword(text, k=k, stopword=stopwords, allowPOS=allowPOS)


def keyphrase(text, k=10, stopword=True):
    """extract keyphrase from text"""

    text = safe_input(text)
    stopwords = SYS_STOPWORDS if stopword else []
    return _summary.keyphrase(text, k=k, stopword=stopwords)


def pinyin(text):
    """get pinyin"""

    text = safe_input(text)
    return _pinyin.translate(text)


def checker(text, level=0):
    """ text checker 文本纠错
    Args:
        level:
            - 0: word
            - 1: doc
    """

    text = safe_input(text)
    return _checker.check(text, level=level)


def sentiment(text, stopword=True):
    """text sentiment analyse"""

    text = safe_input(text)
    stopwords = SYS_STOPWORDS if stopword else []
    return _sentiment.predict(text, stopword=stopwords)


def radical(text):
    """get radical from text"""

    text = safe_input(text)
    return _radical.radical(text)


def tag_mean(tag_name):
    """get meaning of tag"""

    return tag_dict.get(tag_name, '{} not undefined !'.format(tag_name))
