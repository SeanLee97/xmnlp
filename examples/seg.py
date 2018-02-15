# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

descr = """
      中文分词
/ DAG + HMM + Viterbi / 
"""
print(descr)


doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

doc2 = """人工智能: 是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。
人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，
可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。"""

print('\n++++++++++++++++++++++++ usage 1 ++++++++++++++++++++++++\n')

"""
 1. 使用类来进行操作 (推荐)

"""
from xmnlp import XmNLP 

xm = XmNLP(doc)
xm.set_userdict('./userdict.txt')

seg_list1 = xm.seg(hmm=True)
print(' / '.join(seg_list1))
print()

seg_list2 = xm.seg(doc2, hmm=True)
print(' / '.join(seg_list2))



print('\n++++++++++++++++++++++++ usage 2 ++++++++++++++++++++++++\n')

"""
 2. 直接引包使用 (不可使用系统停用词)

"""
import xmnlp

seg_list = xmnlp.seg(doc, hmm=True)
print(' / '.join(seg_list))
