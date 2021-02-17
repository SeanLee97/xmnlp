# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import threading
from typing import List

from xmnlp.config import path as C_PATH
from xmnlp.radical.radical import Radical


model = None
lock = threading.Lock()


def loader():
    """load model"""
    with lock:
        global model
        if model is None:
            print("(Lazy Load) Loading model...")
            model = Radical()
            model.load(C_PATH.radical['model']['radical'])


def radical(text: str) -> List[str]:
    """获取偏旁"""

    loader()
    if not text:
        return None
    return [model.radical(ch) for ch in text]
