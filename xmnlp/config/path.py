# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#
import os

# mode dir
ROOT = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
CORPUS = os.path.join(ROOT, 'corpus')
CHECKER = os.path.join(ROOT, 'checker')
SENTIMENT = os.path.join(ROOT, 'sentiment')
PINYIN = os.path.join(ROOT, 'pinyin')
RADICAL = os.path.join(ROOT, 'radical')


pinyin = {
    'corpus': {
        'pinyin': os.path.join(CORPUS, 'pinyin'),
    },
    'model': {
        'pinyin': os.path.join(PINYIN, 'pinyin.pickle')
    }
}

checker = {
    'corpus': {
        'checker': os.path.join(CHECKER, 'words.txt'),
    },
}


radical = {
    'corpus': {
        'radical': os.path.join(CORPUS, 'radical'),
    },
    'model': {
        'radical': os.path.join(RADICAL, 'radical.pickle')
    }
}

stopword = {
    'corpus': {
        'stopword': os.path.join(ROOT, 'stopword.txt'),
    }
}
