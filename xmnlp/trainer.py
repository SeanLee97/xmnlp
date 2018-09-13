# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

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


from .config import path as C_PATH
from .postag import postag as _postag
from .sentiment.sentiment import Sentiment
from .pinyin.pinyin import Pinyin
from .radical.radical import Radical

import logging

logger = logging.getLogger('xmnlp')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class Trainer(object):

    def __init__(self):
        pass

class PostagTrainer(Trainer):
    @staticmethod
    def hmm(srcfile, outfile):
        logger.info('start to train postag hmm model...')

        postagger = _postag.Postag()
        postagger.train_hmm(srcfile, outfile)

        logger.info('Done !')

    @staticmethod
    def dag(srcfile, outfile):
        logger.info('start to train postag DAG model...')

        postagger = _postag.Postag()
        postagger.train(srcfile, outfile)

        logger.info('Done !')

class SentimentTrainer(Trainer):
    @staticmethod
    def sentiment(posfile, negfile, modelfile):
        logger.info('start to train sentiment model...')

        from . import sys_stopwords
        stm = Sentiment()
        stm.train(posfile, negfile, stopword=sys_stopwords)
        stm.save(modelfile)
        logger.info('Done !')   

class PinyinTrainer(Trainer):
    @staticmethod
    def pinyin(srcfile, outfile):
        logger.info('start to train pinyin model...')
        py = Pinyin()
        py.train(srcfile)
        py.save(outfile)
        logger.info('Done !')

class RadicalTrainer(Trainer):
    @staticmethod
    def radical(srcfile, outfile):
        logger.info('start to train radical model...')
        ra = Radical()
        ra.train(srcfile)
        ra.save(outfile)
        logger.info('Done !')

class SysTrainer(Trainer):
    @staticmethod
    def all():
        logger.info('start to train all model...')
        
        # pinyin
        PinyinTrainer.pinyin(C_PATH.pinyin['corpus']['pinyin'], C_PATH.pinyin['model']['pinyin'])
        
        # sentiment
        SentimentTrainer.sentiment(C_PATH.sentiment['corpus']['pos'], C_PATH.sentiment['corpus']['neg'], C_PATH.sentiment['model']['sentiment'])

        # train postag dag
        PostagTrainer.dag(C_PATH.postag['corpus']['dag'], C_PATH.postag['model']['dag'])
        # train seg hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['seg'], C_PATH.postag['model']['seg'])
        # train tag hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['tag'], C_PATH.postag['model']['tag'])

        # radical
        RadicalTrainer.radical(C_PATH.radical['corpus']['radical'], C_PATH.radical['model']['radical'])

        logger.info('All Done !')
