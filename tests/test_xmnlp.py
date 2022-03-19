# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from functools import partial

import pytest

import xmnlp
from xmnlp.utils import parallel_handler


@pytest.fixture
def lexical_data():
    return ['结婚的和尚未结婚的都成了和尚',
            '我喜欢《瓦尔登湖》这本书，如果你也喜欢，欢迎联系我 xmlee97@gmail.com 一起交流',
            '<h1>谷歌</h1>的网址是https://google.com',
            '现在时间是2021年2月',
            '现任美国总统是拜登']


def lexical_equal(preds, trues):
    for (y_pred, y_true) in zip(preds, trues):
        assert y_pred == y_true


def test_seg(lexical_data):
    lexical_data += ['v', '']
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《', '瓦尔登湖', '》', '这', '本书', '，', '如果', '你', '也', '喜欢', '，', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com', '一起', '交流'],
           ['<', 'h1', '>', '谷歌', '<', '/', 'h1', '>', '的', '网址', '是', 'https', ':', '/', '/', 'google', '.', 'com'],
           ['现在', '时间', '是', '2021', '年', '2', '月'],
           ['现任', '美国', '总统', '是', '拜登'],
	   ['v'],
	   []]
    preds = [xmnlp.seg(data) for data in lexical_data]
    lexical_equal(preds, res)


def test_fast_seg(lexical_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《', '瓦尔登湖', '》', '这', '本书', '，', '如果', '你', '也', '喜欢', '，', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com', '一起', '交流'],
           ['<', 'h1', '>', '谷', '歌', '<', '/', 'h1', '>', '的', '网址', '是', 'https', ':', '/', '/', 'google', '.', 'com'],
           ['现在', '时间', '是', '2021', '年', '2', '月'],
           ['现任', '美国', '总统', '是', '拜登']]
    preds = [xmnlp.fast_seg(data) for data in lexical_data]
    lexical_equal(preds, res)


def test_deep_seg(lexical_data):
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《', '瓦尔登', '湖', '》', '这', '本书', '，', '如果', '你', '也', '喜欢', '，', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com', '一起', '交流'],  # NOQA
           ['<', 'h1', '>', '谷歌', '<', '/', 'h1', '>', '的', '网址', '是', 'https', ':', '/', '/', 'google', '.', 'com'],
           ['现在', '时间', '是', '2021年2月'], ['现任', '美国', '总统', '是', '拜登']]
    preds = [xmnlp.deep_seg(data) for data in lexical_data]
    lexical_equal(preds, res)


def test_seg_parallel(lexical_data):
    seg_parallel = partial(parallel_handler, xmnlp.seg)
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《', '瓦尔登湖', '》', '这', '本书', '，', '如果', '你', '也', '喜欢', '，', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com', '一起', '交流'],
           ['<', 'h1', '>', '谷歌', '<', '/', 'h1', '>', '的', '网址', '是', 'https', ':', '/', '/', 'google', '.', 'com'],
           ['现在', '时间', '是', '2021', '年', '2', '月'],
           ['现任', '美国', '总统', '是', '拜登']]
    preds = seg_parallel(lexical_data)
    lexical_equal(preds, res)


def test_deep_seg_parallel(lexical_data):
    deep_seg_parallel = partial(parallel_handler, xmnlp.deep_seg)
    res = [['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
           ['我', '喜欢', '《', '瓦尔登', '湖', '》', '这', '本书', '，', '如果', '你', '也', '喜欢', '，', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com', '一起', '交流'],  # NOQA
           ['<', 'h1', '>', '谷歌', '<', '/', 'h1', '>', '的', '网址', '是', 'https', ':', '/', '/', 'google', '.', 'com'],
           ['现在', '时间', '是', '2021年2月'], ['现任', '美国', '总统', '是', '拜登']]
    preds = deep_seg_parallel(lexical_data)
    lexical_equal(preds, res)


def test_tag(lexical_data):
    res = [[('结婚', 'v'), ('的', 'u'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'u'), ('都', 'd'), ('成', 'v'), ('了', 'u'), ('和尚', 'nr')],
           [('我', 'r'), ('喜欢', 'v'), ('《', 'x'), ('瓦尔登湖', 'nr'), ('》', 'w'), ('这', 'r'), ('本书', 'r'), ('，', 'x'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('，', 'x'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xmlee97', 'eng'), ('@', 'x'), ('gmail', 'eng'), ('.', 'x'), ('com', 'eng'), ('一起', 'm'), ('交流', 'n')],
           [('<', 'x'), ('h1', 'eng'), ('>', 'w'), ('谷歌', 'nt'), ('<', 'w'), ('/', 'w'), ('h1', 'eng'), ('>', 'w'), ('的', 'u'), ('网址', 'n'), ('是', 'v'), ('https', 'eng'), (':', 'w'), ('/', 'w'), ('/', 'w'), ('google', 'eng'), ('.', 'x'), ('com', 'eng')],
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2021', 'eng'), ('年', 'q'), ('2', 't'), ('月', 'm')],
           [('现任', 'n'), ('美国', 'ns'), ('总统', 'n'), ('是', 'v'), ('拜登', 'nrt')]]
    preds = [xmnlp.tag(data) for data in lexical_data]
    for (y_pred, y_true) in zip(preds, res):
        assert y_pred == y_true


def test_deep_tag(lexical_data):
    res = [[('结婚', 'v'), ('的', 'u'), ('和', 'c'), ('尚未', 'nz'), ('结婚', 'v'), ('的', 'u'), ('都', 'd'), ('成', 'v'), ('了', 'u'), ('和尚', 'nn')],
           [('我', 'r'), ('喜欢', 'v'), ('《', 'w'), ('瓦尔登', 'nr'), ('湖', 'n'), ('》', 'w'), ('这', 'r'), ('本书', 'r'), ('，', 'w'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('，', 'w'), ('欢迎', 'v'), ('联系', 'vn'), ('我', 'r'), ('xmlee97', 'x'), ('@', 'w'), ('gmail', 'x'), ('.', 'w'), ('com', 'x'), ('一起', 's'), ('交流', 'vn')],  # NOQA
           [('<', 'w'), ('h1', 'x'), ('>', 'w'), ('谷歌', 'nt'), ('<', 'w'), ('/', 'w'), ('h1', 'x'), ('>', 'w'), ('的', 'u'), ('网址', 'n'), ('是', 'v'), ('https', 'x'), (':', 'w'), ('/', 'w'), ('/', 'w'), ('google', 'x'), ('.', 'w'), ('com', 'x')],  # NOQA
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2021年2月', 't')],
           [('现任', 'v'), ('美国', 'ns'), ('总统', 'nn'), ('是', 'v'), ('拜登', 'nr')]]
    preds = [xmnlp.deep_tag(data) for data in lexical_data]
    for (y_pred, y_true) in zip(preds, res):
        assert y_pred == y_true


def test_tag_parallel(lexical_data):
    tag_parallel = partial(parallel_handler, xmnlp.tag)
    res = [[('结婚', 'v'), ('的', 'u'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'u'), ('都', 'd'), ('成', 'v'), ('了', 'u'), ('和尚', 'nr')],
           [('我', 'r'), ('喜欢', 'v'), ('《', 'x'), ('瓦尔登湖', 'nr'), ('》', 'w'), ('这', 'r'), ('本书', 'r'), ('，', 'x'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('，', 'x'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xmlee97', 'eng'), ('@', 'x'), ('gmail', 'eng'), ('.', 'x'), ('com', 'eng'), ('一起', 'm'), ('交流', 'n')],
           [('<', 'x'), ('h1', 'eng'), ('>', 'w'), ('谷歌', 'nt'), ('<', 'w'), ('/', 'w'), ('h1', 'eng'), ('>', 'w'), ('的', 'u'), ('网址', 'n'), ('是', 'v'), ('https', 'eng'), (':', 'w'), ('/', 'w'), ('/', 'w'), ('google', 'eng'), ('.', 'x'), ('com', 'eng')],
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2021', 'eng'), ('年', 'q'), ('2', 't'), ('月', 'm')],
           [('现任', 'n'), ('美国', 'ns'), ('总统', 'n'), ('是', 'v'), ('拜登', 'nrt')]]
    preds = tag_parallel(lexical_data)
    lexical_equal(preds, res)


def test_deep_tag_parallel(lexical_data):
    deep_tag_parallel = partial(parallel_handler, xmnlp.deep_tag)
    res = [[('结婚', 'v'), ('的', 'u'), ('和', 'c'), ('尚未', 'nz'), ('结婚', 'v'), ('的', 'u'), ('都', 'd'), ('成', 'v'), ('了', 'u'), ('和尚', 'nn')],
           [('我', 'r'), ('喜欢', 'v'), ('《', 'w'), ('瓦尔登', 'nr'), ('湖', 'n'), ('》', 'w'), ('这', 'r'), ('本书', 'r'), ('，', 'w'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('，', 'w'), ('欢迎', 'v'), ('联系', 'vn'), ('我', 'r'), ('xmlee97', 'x'), ('@', 'w'), ('gmail', 'x'), ('.', 'w'), ('com', 'x'), ('一起', 's'), ('交流', 'vn')],  # NOQA
           [('<', 'w'), ('h1', 'x'), ('>', 'w'), ('谷歌', 'nt'), ('<', 'w'), ('/', 'w'), ('h1', 'x'), ('>', 'w'), ('的', 'u'), ('网址', 'n'), ('是', 'v'), ('https', 'x'), (':', 'w'), ('/', 'w'), ('/', 'w'), ('google', 'x'), ('.', 'w'), ('com', 'x')],  # NOQA
           [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2021年2月', 't')],
           [('现任', 'v'), ('美国', 'ns'), ('总统', 'nn'), ('是', 'v'), ('拜登', 'nr')]]
    preds = deep_tag_parallel(lexical_data)
    lexical_equal(preds, res)


def test_ner(lexical_data):
    res = [[('和尚', 'JOB', 12, 14)],
           [('《瓦尔登湖》', 'BOOK', 3, 9), ('瓦尔登', 'PERSON', 4, 7), ('xmlee97@gmail.com', 'EMAIL', 26, 43)],  # NOQA
           [('谷歌', 'ORGANIZATION', 4, 6), ('https://google.com', 'URL', 15, 33)],
           [('现在', 'TIME', 0, 2), ('2021年2月', 'TIME', 5, 12)],
           [('美国', 'LOCATION', 2, 4), ('总统', 'JOB', 4, 6), ('拜登', 'PERSON', 7, 9)]]
    preds = [xmnlp.ner(data) for data in lexical_data]
    for (y_pred, y_true) in zip(preds, res):
        assert y_pred == y_true


def test_ner_parallel(lexical_data):
    ner_parallel = partial(parallel_handler, xmnlp.ner)
    res = [[('和尚', 'JOB', 12, 14)],
           [('《瓦尔登湖》', 'BOOK', 3, 9), ('瓦尔登', 'PERSON', 4, 7), ('xmlee97@gmail.com', 'EMAIL', 26, 43)],  # NOQA
           [('谷歌', 'ORGANIZATION', 4, 6), ('https://google.com', 'URL', 15, 33)],
           [('现在', 'TIME', 0, 2), ('2021年2月', 'TIME', 5, 12)],
           [('美国', 'LOCATION', 2, 4), ('总统', 'JOB', 4, 6), ('拜登', 'PERSON', 7, 9)]]
    preds = ner_parallel(lexical_data)
    lexical_equal(preds, res)


def test_pinyin():
    assert ['ren', 'gong', 'zhi', 'neng'] == xmnlp.pinyin('人工智能')


def test_pinyin_parallel():
    pinyin_parallel = partial(parallel_handler, xmnlp.pinyin)
    expects = [
        ['ren', 'gong', 'zhi', 'neng'],
        ['Zi', 'ran', 'yu', 'yan', 'chu', 'li']
    ]
    for pred, expected in zip(pinyin_parallel(
        ['人工智能', '自然语言处理']), expects):
        assert pred == expected


def test_radical():
    assert ['自', '灬', '讠', '言', '夂', '王'] == xmnlp.radical('自然语言处理')


def test_radical_parallel():
    radical_parallel = partial(parallel_handler, xmnlp.radical)
    expects = [
        ['人', '工', '日', '月'],
        ['自', '灬', '讠', '言', '夂', '王']
    ]
    for pred, expected in zip(radical_parallel(
        ['人工智能', '自然语言处理']), expects):
        assert pred == expected


def test_sentiment():
    score = xmnlp.sentiment('这酒店真心不错哦')
    assert score[1] > 0.5
    score = xmnlp.sentiment('这酒店真心太差了')
    assert score[1] < 0.5


def test_sentiment_parallel():
    sentiment_parallel = partial(parallel_handler, xmnlp.sentiment)
    scores = list(sentiment_parallel(['这酒店真心不错哦', '这酒店真心太差了']))
    assert scores[0][1] > 0.5
    assert scores[1][1] < 0.5


def test_checker():
    ret = xmnlp.checker('说自己市提前两天排对的。', suggest=False)
    assert ret == [(3, '市'), (9, '对')]


def test_checker_parallel():
    checker_parallel = partial(parallel_handler, xmnlp.checker, suggest=False)
    ret = list(checker_parallel(['说自己市提前两天排对的。', '等啊等，忠于等到了']))
    assert ret == [[(3, '市'), (9, '对')], [(4, '忠')]]


def test_sentence_vector():
    import numpy as np
    from xmnlp.sv import SentenceVector

    query = '我想买手机'
    docs = [
        '我想买苹果手机',
        '我喜欢吃苹果'
    ]

    sv = SentenceVector()
    assert isinstance(sv.similarity(query, docs[0]), np.float32)
    print(sv.most_similar(query, docs))
    assert isinstance(sv.most_similar(query, docs), list)

    sv = SentenceVector(genre='通用')
    assert isinstance(sv.similarity(query, docs[0]), np.float32)
    print(sv.most_similar(query, docs))
    assert isinstance(sv.most_similar(query, docs), list)

    sv = SentenceVector(genre='金融')
    assert isinstance(sv.similarity(query, docs[0]), np.float32)
    print(sv.most_similar(query, docs))
    assert isinstance(sv.most_similar(query, docs), list)

    sv = SentenceVector(genre='国际')
    assert isinstance(sv.similarity(query, docs[0]), np.float32)
    print(sv.most_similar(query, docs))
    assert isinstance(sv.most_similar(query, docs), list)


def test_userdict():
    from xmnlp.lexical.tokenization import Tokenization

    tokenizer = Tokenization('tests/userdict.txt', detect_new_word=False)
    assert tokenizer.seg('冰墩墩和雪容融是北京冬奥会吉祥物') == ['冰墩墩', '和', '雪容融', '是', '北京', '冬奥会', '吉祥物']  # NOQA
    assert tokenizer.tag('冰墩墩和雪容融是北京冬奥会吉祥物') == [('冰墩墩', 'nz'), ('和', 'c'), ('雪容融', 'nz'), ('是', 'v'), ('北京', 'ns'), ('冬奥会', 'j'), ('吉祥物', 'nr')]  # NOQA
