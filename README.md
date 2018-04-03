# xmnlp

小明NLP —— 轻量级中文自然语言处理工具

A Lightweight Chinese Natural Language Processing Toolkit.

# 功能

* 中文分词 & 词性标注
  * 支持繁體
  * 支持自定义词典
* 文本纠错
* 文本摘要 & 关键词提取
* 情感分析
* 文本转拼音

# 安装使用

## 支持环境
```
Linux / Mac os (window 未测试)
python2 / python3(推荐)
```
## 安装
```
git clone https://github.com/SeanLee97/xmnlp.git
cd /path/to/xmnlp
pip install -r requirements.txt
python setup.py install
```

## 依赖库
```
numpy
cPickle #(python2.7)
```

# 算法
* 中文分词：
  * 构建DAG图
  * 动态规划查找，综合正反向（正向加权反向输出）求得DAG最大概率路径
  * 使用了SBME语料训练了一套 HMM + Viterbi 模型，解决未登录词问题
* 文本纠错： bi-gram + levenshtein
* 文本摘要： textrank
* 情感分析： naive bayes
 
# 使用文档
支持两种方式调用
* 通过XmNLP实例对象的方式, 特点：方便使用系统停用词，只需XmNLP(stopword=True) 即可开启
* 通过方法直接调用，特点： 快捷方便

## 分词
### example 1
```python
from xmnlp import XmNLP

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

xm = XmNLP(doc)
xm.set_userdict('/path/to/userdict.txt')
print(xm.seg(hmm=True))

'''
xm  = XmNLP()
xm.set_userdict('/path/to/userdict.txt')
print(xm.seg(doc, hmm=True))
'''
```

### example 2
```python
import xmnlp as xm

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""
xm.set_userdict('/path/to/userdict.txt')
print(xm.seg(doc, hmm=True))
```
result: ['自然语言', '处理', ':', '是', '人工智能', '和', '语言学', '领域', '的', '分支', '学科', '。', '在', '这此', '领域', '中', '探讨', '如何', '处理', '及', '运用', '自然语言', '；', '自然语言', '认知', '则', '是', '指让', '电脑', '“', '懂', '”', '人类', '的', '语言', '。', '自然语言', '生成', '系统', '把', '计算机', '数据', '转化', '为', '自然语言', '。', '自然语言', '理解', '系统', '把', '自然语言', '转化', '为', '计算机程序', '更', '易于', '处理', '的', '形式', '。']

## 词性标注
### example 1
```python
from xmnlp import XmNLP

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

xm = XmNLP(doc)
xm.set_userdict('/path/to/userdict.txt')
print(list(xm.tag()))

'''
xm  = XmNLP()
xm.set_userdict('/path/to/userdict.txt')
print(xm.tag(doc, hmm=True))
'''
```

### example 2
```python
import xmnlp as xm
xm.set_userdict('/path/to/userdict.txt')

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

print(list(xm.tag(doc)))
```
result: [('自然语言', 'l'), ('处理', 'v'), (':', 'un'), ('是', 'v'), ('人工智能', 'n'), ('和', 'c'), ('语言学', 'n'), ('领域', 'n'), ('的', 'uj'), ('分支', 'n'), ('学科', 'n'), ('。', 'un'), ('在', 'p'), ('这此', 'un'), ('领域', 'n'), ('中', 'f'), ('探讨', 'v'), ('如何', 'r'), ('处理', 'v'), ('及', 'c'), ('运用', 'vn'), ('自然语言', 'l'), ('；', 'un'), ('自然语言', 'l'), ('认知', 'v'), ('则', 'd'), ('是', 'v'), ('指让', 'un'), ('电脑', 'n'), ('“', 'un'), ('懂', 'v'), ('”', 'un'), ('人类', 'n'), ('的', 'uj'), ('语言', 'n'), ('。', 'un'), ('自然语言', 'l'), ('生成', 'v'), ('系统', 'n'), ('把', 'p'), ('计算机', 'n'), ('数据', 'n'), ('转化', 'v'), ('为', 'p'), ('自然语言', 'l'), ('。', 'un'), ('自然语言', 'l'), ('理解', 'v'), ('系统', 'n'), ('把', 'p'), ('自然语言', 'l'), ('转化', 'v'), ('为', 'p'), ('计算机程序', 'n'), ('更', 'd'), ('易于', 'v'), ('处理', 'v'), ('的', 'uj'), ('形式', 'n'), ('。', 'un')]

## 文本纠错
### example 1
```python
from xmnlp import XmNLP

doc = """这理风景绣丽，而且天汽不错，我的心情各外舒畅!"""

xm = XmNLP(doc)
print(xm.checker())

'''
xm  = XmNLP()
print(xm.checker(doc))
'''
```

### example 2
```python
import xmnlp as xm

doc = """这理风景绣丽，而且天汽不错，我的心情各外舒畅!"""

print(xm.checker())
```
result: 这里风景秀丽，而且天气不错，我的心情格外舒畅!

## 文本摘要
### example 1
```python
from xmnlp import XmNLP

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

xm  = XmNLP(doc, stopword=True) # stopword=True 使用系统停用词
xm.set_stopword('/path/to/user_stopword.txt') # 使用用户自定义停用词
# keyword
print(xm.keyword())
# keyphrase
pirnt(xm.keyphrase())

'''
xm  = XmNLP()
# keyword
print(xm.keyword(doc))
# keyphrase
pirnt(xm.keyphrase(doc))
'''
```
### example 2
```python
import xmnlp as xm
xm.set_stopword('/path/to/user_stopword.txt') # 使用用户自定义停用词

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

# keyword
print(xm.keyword(doc))
# keyphrase
pirnt(xm.keyphrase(doc))
```
result: 

keyphrase: 

自然语言理解系统自然语言转化计算机程序易于形式

自然语言生成系统计算机数据转化自然语言

自然语言认知指让电脑懂人类语言

这此领域中探讨自然语言

自然语言人工智能语言学领域分支学科

keyword: 
[('自然语言', 2.5960552414414391), ('系统', 1.3424759005594451), ('转化', 1.2404934273839832), ('领域', 1.13500044179745), ('语言', 1.0865431295952139)]

## 情感分析
### example 1
```python
from xmnlp import XmNLP

doc = """这件衣服的质量也太差了吧！一穿就烂！"""

xm  = XmNLP(doc)
print(xm.sentiment())

'''
xm  = XmNLP()
print(xm.sentiment(doc))
'''
```

### example 2
```python
import xmnlp as xm

doc = """这件衣服的质量也太差了吧！一穿就烂！"""
print(xm.sentiment())
```

## 自定义模型
支持用户使用自己的语料训练模型，训练例子在[examples](https://github.com/SeanLee97/xmnlp/tree/master/examples) 的trainer_\*中

## 训练语料
[语料百度网盘](https://pan.baidu.com/s/1UtkaKfNn-R47569VAIeL-A)

## 更多
更多例子请查看[examples](https://github.com/SeanLee97/xmnlp/tree/master/examples)

# Reference:
本项目采用的数据主要有：
* 人民日报语料
* 结巴分词分词数据
* snownlp情感分析语料

本项目受到以下项目的启发
* [jieba](https://github.com/fxsjy/jieba)
* [snownlp](https://github.com/isnowfy/snownlp)

# License
[MIT](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)
