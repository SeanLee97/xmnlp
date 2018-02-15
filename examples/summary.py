# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import print_function

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

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

descr = """
摘要+关键词提取
/ Textrank / 
"""
print(descr)


doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""


print('\n++++++++++++++++++++++++ usage 1 ++++++++++++++++++++++++\n')

"""
 1. 使用类来进行操作 (推荐)

"""
from xmnlp import XmNLP 

xm = XmNLP(doc, stopword=True)
print('keyphrase: ')
for p in xm.keyphrase(k=5):
    print(''.join(p))
    print()
print('keyword: ')
print(xm.keyword(k=5))


print('\n++++++++++++++++++++++++ usage 2 ++++++++++++++++++++++++\n')

import xmnlp
print('keyphrase: ')
for p in xmnlp.keyphrase(doc):
    print(''.join(p))
    print()

print('keyword: ')
print(xmnlp.keyword(doc, k=5))