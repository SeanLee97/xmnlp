# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import unicode_literals
import logging
from .config import path as C_PATH
from .postag import postag as _postag
from .sentiment.sentiment import Sentiment
from .pinyin.pinyin import Pinyin
from .radical.radical import Radical


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

        from . import SYS_STOPWORDS
        stm = Sentiment()
        stm.train(posfile, negfile, stopword=SYS_STOPWORDS)
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
        SentimentTrainer.sentiment(C_PATH.sentiment['corpus']['pos'],
                                   C_PATH.sentiment['corpus']['neg'],
                                   C_PATH.sentiment['model']['sentiment'])

        # train postag dag
        PostagTrainer.dag(C_PATH.postag['corpus']['dag'], C_PATH.postag['model']['dag'])
        # train seg hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['seg'], C_PATH.postag['model']['seg'])
        # train tag hmm
        PostagTrainer.hmm(C_PATH.postag['corpus']['tag'], C_PATH.postag['model']['tag'])

        # radical
        RadicalTrainer.radical(C_PATH.radical['corpus']['radical'],
                               C_PATH.radical['model']['radical'])

        logger.info('All Done !')
