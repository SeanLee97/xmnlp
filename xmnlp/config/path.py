# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#
from __future__ import absolute_import, unicode_literals
import os
import sys
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')

# mode dir
ROOT = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
CORPUS = os.path.join(ROOT, 'corpus')
POSTAG = os.path.join(ROOT, 'postag')
CHECKER = os.path.join(ROOT, 'checker')
SENTIMENT = os.path.join(ROOT, 'sentiment')
PINYIN = os.path.join(ROOT, 'pinyin')
RADICAL = os.path.join(ROOT, 'radical')

# file path funtion
root_fpath = lambda x: os.path.join(ROOT, x)
corpus_fpath = lambda x: os.path.join(CORPUS, x)
postag_fpath = lambda x: os.path.join(POSTAG, x)
checker_fpath = lambda x: os.path.join(CHECKER, x)
sentiment_fpath = lambda x: os.path.join(SENTIMENT, x)
pinyin_fpath = lambda x: os.path.join(PINYIN, x)
radical_fpath = lambda x: os.path.join(RADICAL, x)

postag = {
    'corpus': {
        'dag': corpus_fpath('word_freq_tag'),
        'seg': corpus_fpath('seg'),
        'tag': corpus_fpath('tag')
    },
    'model': {
        'dag': postag_fpath('postag.pickle'),
        'seg': postag_fpath('seg.hmm.pickle'),
        'tag': postag_fpath('tag.hmm.pickle')
    }
}

pinyin = {
    'corpus': {
        'pinyin': corpus_fpath('pinyin'),
    },
    'model': {
        'pinyin': pinyin_fpath('pinyin.pickle')
    }
}

checker = {
    'corpus': {
        'checker': checker_fpath('words.txt'),
    },
}

sentiment = {
    'corpus': {
        'pos': corpus_fpath(os.path.join('sentiment', 'pos.txt')),
        'neg': corpus_fpath(os.path.join('sentiment', 'neg.txt')),
    },
    'model': {
        'sentiment': sentiment_fpath('sentiment.pickle')
    }
}

radical = {
    'corpus': {
        'radical': corpus_fpath('radical'),
    },
    'model': {
        'radical': radical_fpath('radical.pickle')
    }
}

stopword = {
    'corpus': {
        'stopword': root_fpath('stopword.txt'),
    }
}
