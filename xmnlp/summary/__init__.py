# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from typing import List, Tuple, Optional

import xmnlp
from xmnlp.summary.textrank import TextRank, KeywordTextRank
from xmnlp.utils import split_text


def keyword(text: str,
            k: int = 10,
            stopword: Optional[List[str]] = None,
            allowPOS: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """关键词抽取
    Args:
      text: str, 输入文本
      k: int, 返回 topk 个关键词
      stopword: List[str], 关键词列表，默认 None
      allowPos: List[str], 允许的词性
    Return:
      List[Tuple[str, float]], e.g., [('word', weight), ...]
    """
    text = text.strip()
    if stopword is None:
        stopword = []
    if allowPOS is None:
        allowPOS = []
    words = []
    for ret in xmnlp.tag_parallel(split_text(text)):
        for w, t in ret:
            if w not in stopword and t in allowPOS:
                words.append(w)

    words = KeywordTextRank(words)
    return words.topk(k)


def keyphrase(text: str,
              k: int = 10,
              stopword: Optional[List[str]] = None) -> List[str]:
    """关键句抽取
    Args:
      text: str, 输入文本
      k: int, 返回 topk 个关键词
      stopword: List[str], 关键词列表，默认 None
    Return:
      List[str]
    """
    text = text.strip()
    if stopword is None:
        stopword = []

    docs = []
    for sent in split_text(text):
        words = []
        for word, _ in xmnlp.tag(sent):
            if word not in stopword:
                words.append(word)
        docs.append(words)

    tr = TextRank(docs)
    res = []
    for idx in tr.topk(k):
        res.append(''.join(docs[idx]))

    return res
