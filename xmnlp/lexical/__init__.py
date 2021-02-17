# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import os
import re
import threading
from typing import List, Tuple

from xmnlp import config
from xmnlp.lexical.lexical_model import LexicalDecoder


NER_RULES = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9\-_\.]+@[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_\.]+", re.IGNORECASE)),
    ("URL", re.compile(r"(?:http|ftp)s?://"  # http:// or https://
                       r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{1,}\.?)|" # domain
                       r"localhost|"  # localhost
                       r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or ip
                       r"(?::\d+)?"  # optional port
                       r"(?:/?|[/?]\S+)[a-zA-Z0-9/]+", re.IGNORECASE)),
    ("BOOK", re.compile(r"《.+》"))
]
TAG_MAP = {
    'ORGANIZATION': 'nt',
    'TIME': 't',
    'JOB': 'nn',
    'PERSON': 'nr',
    'LOCATION': 'ns'
}
lexical = None
lock = threading.Lock()


def load_lexical(reload: bool = False) -> None:
    with lock:
        global lexical
        if lexical is None or reload:
            if config.MODEL_DIR is None:
                raise ValueError("Error: 模型地址未设置，请根据文档「安装」 -> 「下载模型」指引下载并配置模型。")

            print('Lazy load lexical...')
            lexical = LexicalDecoder(
                os.path.join(config.MODEL_DIR, 'lexical'), starts=[0], ends=[0])


def seg(doc: str) -> List[str]:
    """分词
    """
    load_lexical()
    doc = doc.strip()
    return [w for w, _ in lexical.predict(doc)]


def tag(doc: str) -> List[Tuple[str, str]]:
    """词性标注
    Args:
      doc: str
    Return: List[Tuple[str, str]], e.g., [(word, tag), ...]
    """
    load_lexical()
    return [(w, TAG_MAP.get(t, t)) for w, t in lexical.predict(doc)]


def ner(doc: str) -> List[Tuple[str, str, int, int]]:
    """命名体识别
    支持的实体类型为:
      (ORGANIZATION, 机构), (TIME, 时间), (JOB, 职业), (PERSON, 人名), (LOCATION, 地点),
      (EMAIL, 邮件), (URL, 网址), (BOOK, 书)

    Args:
      doc: str
    Return: List[Tuple[str, str, int, int]], e.g., [(word, tag, start index, end index), ...]
    """
    load_lexical()
    doc = doc.strip()
    ret = []
    # rule
    for (mathches, tag) in [(r[1].finditer(doc), r[0]) for r in NER_RULES]:
            for mathched in mathches:
                span = mathched.span()
                ret.append((doc[span[0]: span[1]], tag, span[0], span[1]))
    # model
    start_position = 0
    for w, t in lexical.predict(doc):
        if t in TAG_MAP:
            ret.append((w, t, start_position, start_position + len(w)))
        start_position += len(w)
    
    ret.sort(key=lambda x: (x[2], x[3]))
    return ret
