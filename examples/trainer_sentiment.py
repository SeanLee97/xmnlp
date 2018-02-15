# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

"""MIT License
Copyright (c) 2018 Sean
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import sys
sys.path.append("..")

from xmnlp.trainer import SentimentTrainer

"""
 1. 使用用户自定义语料训练模型

"""
SentimentTrainer.sentiment('./corpus/sentiment/pos.txt', './corpus/sentiment/neg.txt', './models/sentiment.pickle')

"""
 2. 使用自定义模型
"""
from xmnlp.sentiment import load, predict

# load
load('./models/sentiment.pickle')

doc = """这件衣服的质量也太差了吧！一穿就烂！"""
doc2 = """天气太好了，我们去钓鱼吧"""

print('Text: ', doc)
print('Score: ', predict(doc))
print('Text: ', doc2)
print('Score: ', predict(doc2))