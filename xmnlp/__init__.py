# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import os
import concurrent.futures as futures
from functools import partial
from typing import (
    List, Tuple, Generator, Optional, Union, Dict
)

from xmnlp import config
from xmnlp import lexical
from xmnlp import summary
from xmnlp import checker as _checker
from xmnlp import sentiment as _sentiment
from xmnlp import pinyin as _pinyin
from xmnlp import radical as _radical
from xmnlp.utils import load_stopword


__author__ = 'Sean Lee <xmlee97@gmail.com>'
__version__ = '0.3.0'


seg = lexical.seg
tag = lexical.tag
ner = lexical.ner
pinyin = _pinyin.translate
radical = _radical.radical
checker = _checker.spellcheck
sentiment = _sentiment.sentiment


def set_model(dirname: str) -> None:
    config.MODEL_DIR = dirname


def set_stopword(fpath: str) -> None:
    """set stopwords from file"""
    config.SYS_STOPWORDS = set(load_stopword(fpath)) | config.SYS_STOPWORDS


def seg_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[str], None, None]:
    """seg parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[List[str]]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for words in executor.map(seg, texts):
            yield words


def tag_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[Tuple[str, str]], None, None]:
    """tag parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[List[Tuple[str, str]]]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(tag, texts):
            yield ret


def ner_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[Tuple[str, str, int, int]], None, None]:
    """ner parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[List[Tuple[str, str]]]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(ner, texts):
            yield ret


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


def keyword_parallel(texts: List[str],
                     k: int = 10,
                     stopword: bool = True,
                     allowPOS: Optional[List[str]] = None,
                     n_jobs: int = 2) -> Generator[List[Tuple[str, float]], None, None]:
    """keyword parallel
    Args:
      texts: List[str]
      k: int, 返回 topk 个关键词
      stopword: bool, 是否设置停用词
      allowPOS: Optional[List[str]], 允许的词性, 默认为系统自定义的词性列表
      n_jobs: int, pool size of threads
    Return:
      List[Tuple[str, float]]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    keyword_func = partial(keyword, k=k, stopword=stopword, allowPOS=allowPOS)
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(keyword_func, texts):
            yield ret


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


def keyphrase_parallel(texts: List[str],
                       k: int = 10,
                       stopword: bool = False,
                       n_jobs: int = 2) -> Generator[List[str], None, None]:
    """keyphrase parallel
    Args:
      texts: List[str]
      k: int, 返回 topk 个关键句
      stopword: bool, 是否设置停用词
      n_jobs: int, pool size of threads
    Return:
      List[Tuple[str, float]]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    keyphrase_func = partial(keyphrase, k=k, stopword=stopword)
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(keyphrase_func, texts):
            yield ret


def sentiment_parallel(texts: List[str], n_jobs=2) -> Generator[Tuple[float, float], None, None]:
    """sentiment parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[Tuple[float, float], None, None]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(sentiment, texts):
            yield ret


def pinyin_parallel(texts: List[str], n_jobs=2) -> Generator[List[str], None, None]:
    """pinyin parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[List[str], None, None]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(pinyin, texts):
            yield ret


def radical_parallel(texts: List[str], n_jobs=2) -> Generator[List[str], None, None]:
    """radical parallel
    Args:
      texts: List[str]
      n_jobs: int, pool size of threads
    Return:
      Generator[List[str], None, None]
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(radical, texts):
            yield ret


def checker_parallel(texts: List[str],
                     suggest: bool = True,
                     k: int = 5,
                     max_k: int = 200,
                     n_jobs: int = 2) -> Generator[
                        Union[List[Tuple[int, str]],
                              Dict[Tuple[int, str], List[Tuple[str, float]]]
                        ], None, None]:
    """checker parallel
    Args:
      texts: List[str]
      suggest: bool, 是否返回建议词
      k: int, 返回 topk 个建议词
      max_k: int, 拼音相同词最大检索次数
      n_jobs: int, pool size of threads
    Return:
      Generator
    """
    if not isinstance(texts, list):
        raise ValueError("You should pass a list of texts")
    checker_func = partial(checker, suggest=suggest, k=k, max_k=max_k)
    with futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for ret in executor.map(checker_func, texts):
            yield ret
