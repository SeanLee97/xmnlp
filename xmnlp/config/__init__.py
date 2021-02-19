# -*- coding: utf-8 -*-

import os

from xmnlp.config import path
from xmnlp.utils import load_stopword


# 模型地址配置
MODEL_DIR = os.getenv('XMNLP_MODEL', None)
ALLOW_POS = ['an', 'i', 'j', 'l', 'n', 'nr', 'ns', 'nt', 'nz',
             't', 'v', 'vd', 'vn', 'x', 'nn', 'g']
SYS_STOPWORDS = load_stopword(path.stopword['corpus']['stopword'])
