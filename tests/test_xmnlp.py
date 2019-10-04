# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

import pytest
import xmnlp


@pytest.fixture
def postag_data():
    return ['结婚的和尚未结婚的都成了和尚',
            '我喜欢《瓦尔登湖》这本书，如果你也喜欢欢迎联系我 xxx@gmail.com',
            '<h1>谷歌</h1>的网址是https://google.com',
            '现在时间是2019年10月']


def postag_equal(preds, trues):
    for (y_pred, y_true) in zip(preds, trues):
        assert y_pred == y_true


def test_seg(postag_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《瓦尔登湖》', '这', '本书', '，', '如果', '你', '也', '喜欢', '欢迎', '联系', '我', 'xxx@gmail.com'],
           ['<h1>', '谷歌', '</h1>', '的', '网址', '是', 'https://google.com'],
           ['现在', '时间', '是', '2019年10月']]
    preds = [xmnlp.seg(data) for data in postag_data]
    postag_equal(preds, res)


def test_seg_parallel(postag_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《瓦尔登湖》', '这', '本书', '，', '如果', '你', '也', '喜欢', '欢迎', '联系', '我', 'xxx@gmail.com'],
           ['<h1>', '谷歌', '</h1>', '的', '网址', '是', 'https://google.com'],
           ['现在', '时间', '是', '2019年10月']]
    preds = xmnlp.seg_parallel(postag_data)
    postag_equal(preds, res)


def test_tag(postag_data):
    res = [[('结婚', 'v'), ('的', 'uj'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'uj'), ('都', 'd'), ('成', 'n'), ('了', 'ul'), ('和尚', 'nr')],
           [('我', 'r'), ('喜欢', 'v'), ('《瓦尔登湖》', 'book'), ('这', 'r'), ('本书', 'r'), ('，', 'w'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xxx@gmail.com', 'email')],
           [('<h1>', 'html'), ('谷歌', 'n'), ('</h1>', 'html'), ('的', 'uj'), ('网址', 'n'), ('是', 'v'), ('https://google.com', 'url')],
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2019年10月', 'datetime')]]
    preds = [xmnlp.tag(data) for data in postag_data]
    for (y_pred, y_true) in zip(preds, res):
        assert y_pred == y_true


def test_tag_parallel(postag_data):
    res = [[('结婚', 'v'), ('的', 'uj'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'uj'), ('都', 'd'), ('成', 'n'), ('了', 'ul'), ('和尚', 'nr')],
           [('我', 'r'), ('喜欢', 'v'), ('《瓦尔登湖》', 'book'), ('这', 'r'), ('本书', 'r'), ('，', 'w'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xxx@gmail.com', 'email')],
           [('<h1>', 'html'), ('谷歌', 'n'), ('</h1>', 'html'), ('的', 'uj'), ('网址', 'n'), ('是', 'v'), ('https://google.com', 'url')],
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2019年10月', 'datetime')]]
    preds = xmnlp.tag_parallel(postag_data)
    postag_equal(preds, res)


def test_pinyin():
    assert ['ren', 'gong', 'zhi', 'neng'] == xmnlp.pinyin('人工智能')


def test_radical():
    assert ['自', '灬', '讠', '言', '夂', '王'] == xmnlp.radical('自然语言处理')


def test_sentiment():
    score = xmnlp.sentiment('这酒店真心不错')
    assert score > 0.5
