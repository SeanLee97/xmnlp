# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import os
import threading
from typing import Tuple

from xmnlp import config
from xmnlp.sentiment.sentiment_model import SentimentModel


sentiment_model = None
lock = threading.Lock()


def load_sentiment(reload: bool = False) -> None:
    with lock:
        global sentiment_model
        if sentiment_model is None or reload:
            if config.MODEL_DIR is None:
                raise ValueError("Error: 模型地址未设置，请根据文档「安装」 -> 「下载模型」指引下载并配置模型。")

            print('Lazy load sentiment...')
            sentiment_model = SentimentModel(
                os.path.join(config.MODEL_DIR, 'sentiment'))


def sentiment(doc: str) -> Tuple[float, float]:
    """ 情感分类
    Args:
      doc: str
    Return:
      Tuple[float, float], [proba of negative, proba of postive]
    """
    load_sentiment()
    doc = doc.strip()
    return sentiment_model.predict(doc)
