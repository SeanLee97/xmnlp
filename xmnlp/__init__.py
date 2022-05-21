# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

__author__ = 'Sean Lee <xmlee97@gmail.com>'
__version__ = '0.5.2'


from typing import List, Tuple, Optional

from xmnlp import config
from xmnlp import lexical
from xmnlp import summary
from xmnlp import checker as _checker
from xmnlp import sentiment as _sentiment
from xmnlp import pinyin as _pinyin
from xmnlp import radical as _radical
from xmnlp.utils import load_stopword


seg = lexical.seg
tag = lexical.tag
fast_seg = lexical.fast_seg
fast_tag = lexical.fast_tag
deep_seg = lexical.deep_seg
deep_tag = lexical.deep_tag
ner = lexical.ner

pinyin = _pinyin.translate
radical = _radical.radical
checker = _checker.spellcheck
sentiment = _sentiment.sentiment


def set_model(dirname: str):
    config.MODEL_DIR = dirname


def set_stopword(fpath: str):
    """set stopwords from file"""
    config.SYS_STOPWORDS = set(load_stopword(fpath)) | config.SYS_STOPWORDS


def keyword(text: str,
            k: int = 10,
            stopword: bool = True,
            allowPOS: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """extract keyword from text
    Args:
      text: str
      k: int, 返回 topk 个关键词
      stopword: bool, 是否设置停用词
      allowPOS: Optional[List[str]], 允许的词性，默认为系统自定义的词性列表
    Return:
      List[Tuple[str, float]]
    """
    if allowPOS is None:
        allowPOS = config.ALLOW_POS
    stopwords = config.SYS_STOPWORDS if stopword else []

    return summary.keyword(text, k=k, stopword=stopwords, allowPOS=allowPOS)


def keyphrase(text: str,
              k: int = 10,
              stopword: bool = False) -> List[str]:
    """keyphrase extraction
    Args:
      text: str
      k: int, 返回 topk 个关键句
      stopword: bool, 是否设置停用词
      allowPOS: Optional[List[str]], 允许的词性, 默认为系统自定义的词性列表
    Return:
      List[str]
    """
    stopwords = config.SYS_STOPWORDS if stopword else []
    return summary.keyphrase(text, k=k, stopword=stopwords)
