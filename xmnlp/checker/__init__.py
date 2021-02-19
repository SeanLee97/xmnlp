# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import os
import threading
from typing import Union, List, Tuple, Dict

from xmnlp import config
from xmnlp.checker.checker import CheckerDecoder


lock = threading.Lock()
checker = None


def load_checker(reload: bool = False) -> None:
    with lock:
        global checker
        if checker is None or reload:
            if config.MODEL_DIR is None:
                raise ValueError("Error: 模型地址未设置，请根据文档「安装」 -> 「下载模型」指引下载并配置模型。")

            print('Lazy load checker...')
            checker = CheckerDecoder(
                os.path.join(config.MODEL_DIR, 'checker'))


def spellcheck(text: str,
               suggest: bool = True,
               k: int = 5,
               max_k: int = 200) -> Union[
                   List[Tuple[int, str]],
                   Dict[Tuple[int, str], List[Tuple[str, float]]]]:
    """spell check
    """
    load_checker()
    return checker.predict(text, suggest=suggest, k=k, max_k=max_k)
