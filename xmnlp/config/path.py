# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

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
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

import os

_root_ = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])

_corpus_ = os.path.join(_root_, 'corpus')
_postag_ = os.path.join(_root_, 'postag')
_checker_ = os.path.join(_root_, 'checker')
_sentiment_ = os.path.join(_root_, 'sentiment')
_pinyin_ = os.path.join(_root_, 'pinyin')

root_f = lambda x: os.path.join(_root_, x)
corpus_f = lambda x: os.path.join(_corpus_, x)
postag_f = lambda x: os.path.join(_postag_, x)
checker_f = lambda x: os.path.join(_checker_, x)
sentiment_f = lambda x: os.path.join(_sentiment_, x)
pinyin_f = lambda x: os.path.join(_pinyin_, x)

postag = {
    'corpus': {
        'dag': corpus_f('word_freq_tag'),
        'seg': corpus_f('seg'),
        'tag': corpus_f('tag')
    },
    'model': {
        'dag': postag_f('postag.pickle'),
        'seg': postag_f('seg.hmm.pickle'),
        'tag': postag_f('tag.hmm.pickle')
    }
}

pinyin = {
    'corpus': {
        'pinyin': corpus_f('pinyin'),
    },
    'model': {
        'pinyin': pinyin_f('pinyin.pickle')
    }
}

checker = {
    'corpus': {
        'checker': corpus_f('checker'),
    },
    'model': {
        'checker': checker_f('checker.pickle')
    }
}

sentiment = {
    'corpus': {
        'pos': corpus_f(os.path.join('sentiment', 'pos.txt')),
        'neg': corpus_f(os.path.join('sentiment', 'neg.txt')),
    },
    'model': {
        'sentiment': sentiment_f('sentiment.pickle')
    }
}

stopword = {
    'corpus': {
        'stopword': root_f('stopword.txt'),
    }
}
