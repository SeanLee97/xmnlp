# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import re
import threading
from typing import List

from xmnlp.pinyin.pinyin import Pinyin
from xmnlp.config import path as C_PATH


zh = re.compile(r"([\u4E00-\u9FA5]+)", flags=re.UNICODE)
lock = threading.Lock()
model = None


def loader():
    """load model"""
    with lock:
        global model
        if model is None:
            print("(Lazy Load) Loading model...")
            model = Pinyin()
            model.load(C_PATH.pinyin['model']['pinyin'])


def translate(text: str) -> List[str]:
    """translate chinese to pinyin"""
    loader()
    ret = []
    for s in zh.split(text):
        s = s.strip()
        if not s:
            continue
        if zh.match(s):
            ret += model.translate(s)
        else:
            for word in s.split():
                word = word.strip()
                if word:
                    ret.append(word)
    return ret
