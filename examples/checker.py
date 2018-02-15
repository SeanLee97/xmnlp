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
      文本纠错
/ n-gam + levenshtein / 
"""
print(descr)


doc = """这理风景绣丽，而且天汽不错，我的心情各外舒畅!"""

print('\n++++++++++++++++++++++++ usage 1 ++++++++++++++++++++++++\n')

"""
 1. 使用类来进行操作

"""
from xmnlp import XmNLP 
xm = XmNLP(doc)
print('Error: \n', doc)
print('Correct: \n', xm.checker())


print('\n++++++++++++++++++++++++ usage 2 ++++++++++++++++++++++++\n')

import xmnlp
print('Error: \n', doc)
print('Correct: \n', xmnlp.checker(doc))