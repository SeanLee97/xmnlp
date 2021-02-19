# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import logging
from xmnlp.config import path as C_PATH
from xmnlp.pinyin.pinyin import Pinyin
from xmnlp.radical.radical import Radical


logger = logging.getLogger('xmnlp')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Trainer:
    def __init__(self):
        pass


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

        # radical
        RadicalTrainer.radical(C_PATH.radical['corpus']['radical'],
                               C_PATH.radical['model']['radical'])

        logger.info('All Done !')
