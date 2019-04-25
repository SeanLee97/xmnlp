# -*- coding: utf-8 -*-

import xmnlp
import pytest


@pytest.fixture
def postag_data():
    return ['结婚的和尚未结婚的都成了和尚',
            '工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作',
            '他正在量和服尺寸']


def postag_equal(preds, trues):
    for (y_pred, y_true) in zip(preds, trues):
        assert y_pred == y_true


def test_seg(postag_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
            ['工信处', '女干事', '每月', '经过', '下属', '科室', '都', '要', '亲口', '交代', '24', '口', '交换机', '等', '技术性', '器件', '的', '安装', '工作'],
            ['他', '正在', '量', '和服', '尺寸']]
    preds = [xmnlp.seg(data) for data in postag_data]
    postag_equal(preds, res)


def test_seg_parallel(postag_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
            ['工信处', '女干事', '每月', '经过', '下属', '科室', '都', '要', '亲口', '交代', '24', '口', '交换机', '等', '技术性', '器件', '的', '安装', '工作'],
            ['他', '正在', '量', '和服', '尺寸']]
    preds = xmnlp.seg_parallel(postag_data)
    postag_equal(preds, res)


def test_tag(postag_data):
    res = [[('结婚', 'v'), ('的', 'uj'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'uj'), ('都', 'd'), ('成', 'n'), ('了', 'ul'), ('和尚', 'nr')],
           [('工信处', 'n'), ('女干事', 'n'), ('每月', 'r'), ('经过', 'p'), ('下属', 'v'), ('科室', 'n'), ('都', 'd'), ('要', 'v'), ('亲口', 'n'), ('交代', 'n'), ('24', 'm'), ('口', 'q'), ('交换机', 'n'), ('等', 'u'), ('技术性', 'n'), ('器件', 'n'), ('的', 'uj'), ('安装', 'v'), ('工作', 'vn')],
           [('他', 'r'), ('正在', 't'), ('量', 'n'), ('和服', 'nz'), ('尺寸', 'n')]]
    preds = [xmnlp.tag(data) for data in postag_data]
    for (y_pred, y_true) in zip(preds, res):
        assert y_pred == y_true


def test_tag_parallel(postag_data):
    res = [[('结婚', 'v'), ('的', 'uj'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'uj'), ('都', 'd'), ('成', 'n'), ('了', 'ul'), ('和尚', 'nr')],
           [('工信处', 'n'), ('女干事', 'n'), ('每月', 'r'), ('经过', 'p'), ('下属', 'v'), ('科室', 'n'), ('都', 'd'), ('要', 'v'), ('亲口', 'n'), ('交代', 'n'), ('24', 'm'), ('口', 'q'), ('交换机', 'n'), ('等', 'u'), ('技术性', 'n'), ('器件', 'n'), ('的', 'uj'), ('安装', 'v'), ('工作', 'vn')],
           [('他', 'r'), ('正在', 't'), ('量', 'n'), ('和服', 'nz'), ('尺寸', 'n')]]
    preds = xmnlp.tag_parallel(postag_data)
    postag_equal(preds, res)


def test_pinyin():
    assert ['ren', 'gong', 'zhi', 'neng'] == xmnlp.pinyin('人工智能')


def test_radical():
    assert ['自', '灬', '讠', '言', '夂', '王'] == xmnlp.radical('自然语言处理')


def test_sentiment():
    score = xmnlp.sentiment('这酒店真心不错')
    assert score > 0.5
