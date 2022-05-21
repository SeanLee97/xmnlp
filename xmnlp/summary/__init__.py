# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from typing import List, Tuple, Optional

from xmnlp.lexical import tag
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
    for text in split_text(text):
        for w, t in tag(text):
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
        for word, _ in tag(sent):
            if word not in stopword:
                words.append(word)
        docs.append(words)

    tr = TextRank(docs)
    res = []
    for idx in tr.topk(k):
        res.append(''.join(docs[idx]))

    return res
