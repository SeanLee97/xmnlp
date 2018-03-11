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


from .postag import postag
from .checker import checker
from .sentiment import sentiment
from .pinyin import pinyin
import logging

logger = logging.getLogger('/ xmnlp / ')
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

        postagger = postag.Postag()
        postagger.train_hmm(srcfile, outfile)

        logger.info('Done !')

    @staticmethod
    def dag(srcfile, outfile):
        logger.info('start to train postag DAG model...')

        postagger = postag.Postag()
        postagger.train(srcfile, outfile)

        logger.info('Done !')

class CheckerTrainer(Trainer):
    @staticmethod
    def checker(srcfile, outfile):
        logger.info('start to train checker model...')

        chkr = checker.Checker()
        chkr.train(srcfile)
        chkr.save(outfile)
        logger.info('Done !')

class SentimentTrainer(Trainer):
    @staticmethod
    def sentiment(posfile, negfile, modelfile):

        logger.info('start to train sentiment model...')

        stm = sentiment.Sentiment()
        stm.train(posfile, negfile)
        stm.save(modelfile)
        logger.info('Done !')   

class PinyinTrainer(Trainer):
    @staticmethod
    def pinyin(srcfile, outfile):
        logger.info('start to train pinyin model...')
        py = pinyin.Pinyin()
        py.train(srcfile)
        py.save(outfile)
        logger.info('Done !')


class SysTrainer(Trainer):
    @staticmethod
    def all():
        logger.info('start to train all model...')
        from .config import path as C_PATH
        
        # pinyin
        PinyinTrainer.pinyin(C_PATH.pinyin['corpus']['pinyin'], C_PATH.pinyin['model']['pinyin'])
        
        # train postag dag
        PostagTrainer.dag(C_PATH.postag['corpus']['dag'], C_PATH.postag['model']['dag'])
        # train seg hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['seg'], C_PATH.postag['model']['seg'])
        # train tag hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['tag'], C_PATH.postag['model']['tag'])
        
        # checker
        CheckerTrainer.checker(C_PATH.checker['corpus']['checker'], C_PATH.checker['model']['checker'])

        # sentiment
        SentimentTrainer.sentiment(C_PATH.sentiment['corpus']['pos'], C_PATH.sentiment['corpus']['neg'], C_PATH.sentiment['model']['sentiment'])
        
        logger.info('All Done !')
