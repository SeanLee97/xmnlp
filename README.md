<p align='center'>/ xmnlp /</p>
<p align='center'>小明NLP — 轻量级中文自然语言处理工具</p>
<p align='center'> A Lightweight Chinese Natural Language Processing Toolkit</p>
<p align='center'>v 0.1.8</p>

[![pypi](https://img.shields.io/badge/pypi-v0.1.8-blue.svg)](https://pypi.org/project/xmnlp/)
![python version](https://img.shields.io/badge/python-2%2C3-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
[![GitHub license](https://img.shields.io/github/license/SeanLee97/xmnlp.svg)](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)


## 功能概览

* 中文分词 & 词性标注
  * 支持繁體
  * 支持自定义词典
* 中文拼写检查
* 文本摘要 & 关键词提取
* 情感分析
* 文本转拼音
* 获取汉字偏旁部首

## 安装使用

### 安装方式
#### 方式一
```
pip install xmnlp
```

#### 方式二
```
git clone https://github.com/SeanLee97/xmnlp.git
cd /path/to/xmnlp
pip install -r requirements.txt
python setup.py install
```

### 依赖库
```
numpy
cPickle #(python2.7)
```
## 使用文档
### 中文分词
```python
import xmnlp
xmnlp.set_userdict('/path/to/userdict.txt')

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

print(xmnlp.seg(doc, hmm=True))
```
结果输出
```
['自然语言', '处理', ':', '是', '人工智能', '和', '语言学', '领域', '的', '分支', '学科', '。', '在', '这此', '领域', '中', '探讨', '如何', '处理', '及', '运用', '自然语言', '；', '自然语言', '认知', '则', '是', '指让', '电脑', '“', '懂', '”', '人类', '的', '语言', '。', '自然语言', '生成', '系统', '把', '计算机', '数据', '转化', '为', '自然语言', '。', '自然语言', '理解', '系统', '把', '自然语言', '转化', '为', '计算机程序', '更', '易于', '处理', '的', '形式', '。']
```

### 词性标注

```python
import xmnlp
xm.set_userdict('/path/to/userdict.txt')

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

print(list(xm.tag(doc)))
```

结果输出

```
[('自然语言', 'l'), ('处理', 'v'), (':', 'un'), ('是', 'v'), ('人工智能', 'n'), ('和', 'c'), ('语言学', 'n'), ('领域', 'n'), ('的', 'uj'), ('分支', 'n'), ('学科', 'n'), ('。', 'un'), ('在', 'p'), ('这此', 'un'), ('领域', 'n'), ('中', 'f'), ('探讨', 'v'), ('如何', 'r'), ('处理', 'v'), ('及', 'c'), ('运用', 'vn'), ('自然语言', 'l'), ('；', 'un'), ('自然语言', 'l'), ('认知', 'v'), ('则', 'd'), ('是', 'v'), ('指让', 'un'), ('电脑', 'n'), ('“', 'un'), ('懂', 'v'), ('”', 'un'), ('人类', 'n'), ('的', 'uj'), ('语言', 'n'), ('。', 'un'), ('自然语言', 'l'), ('生成', 'v'), ('系统', 'n'), ('把', 'p'), ('计算机', 'n'), ('数据', 'n'), ('转化', 'v'), ('为', 'p'), ('自然语言', 'l'), ('。', 'un'), ('自然语言', 'l'), ('理解', 'v'), ('系统', 'n'), ('把', 'p'), ('自然语言', 'l'), ('转化', 'v'), ('为', 'p'), ('计算机程序', 'n'), ('更', 'd'), ('易于', 'v'), ('处理', 'v'), ('的', 'uj'), ('形式', 'n'), ('。', 'un')]
```

### 拼写检查
此功能基于symspell实现，建议用来检查词级别的错误，对于句子尚未能很好的解决拼写错误问题，**第一次加载字典的速度较慢（由词典大小决定）**

#### 词级别
```python
import xmnlp
xmnlp.set_userdict('./userdict.txt')

doc = """中国人敏共和国"""

print('Error: \n', doc)
ret = xmnlp.checker(doc, level=0) # level = 0
print('Correct: \n', ret)
```

结果输出
```
中华人民共和国
```

#### 句子级别
`level=1`, 不建议使用，句级别仅返回所有可能的拼写检查结果

```python
import xmnlp
xmnlp.set_userdict('./userdict.txt')

doc = """今天天汽不错哦"""

print('Error: \n', doc)
ret = xmnlp.checker(doc, level=1) # level = 1
print('Correct: \n', ret)
```

结果输出

```
['今天天气', '不错']
```

### 文本摘要
基于`textrank`算法实现

```python
import xmnlp
xm.set_stopword('/path/to/stopword.txt') # 添加用户自定义停用词

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

# keyword
print(xm.keyword(doc))
# keyphrase
print(xm.keyphrase(doc))
```

结果输出

```
keyphrase: 

自然语言理解系统自然语言转化计算机程序易于形式
自然语言生成系统计算机数据转化自然语言
自然语言认知指让电脑懂人类语言
这此领域中探讨自然语言
自然语言人工智能语言学领域分支学科

keyword: 
[('自然语言', 2.5960552414414391), ('系统', 1.3424759005594451), ('转化', 1.2404934273839832), ('领域', 1.13500044179745), ('语言', 1.0865431295952139)]
```

### 情感分析
基于朴素贝叶斯算法实现，基于酒店评价数据训练
```python
import xmnlp
xmnlp.set_stopword('/path/to/stopword.txt') # 用户自定义停用词

doc = """这件衣服的质量也太差了吧！"""
doc2 = """这酒店真心不错"""
print('Text: ', doc)
print('Score: ', xmnlp.sentiment(doc))
print('Text: ', doc2)
print('Score: ', xmnlp.sentiment(doc2))
```

结果输出
```
Text:  这件衣服的质量也太差了吧！
Score:  0.09661951767426591
Text:  这酒店真心不错
Score:  0.7947237609561072
```
### 汉字部首
部首是一种文本的特征，在深度学习中我们有时可以加入部首特征来训练网络

```python
import xmnlp
print(xmnlp.radical('自然语言处理'))
```
结果输出
```
['自', '灬', '讠', '言', '夂', '王']
```

### 自定义模型
支持用户使用自己的语料训练模型，训练例子在[examples](https://github.com/SeanLee97/xmnlp/tree/master/examples) 的trainer_\*中

### 训练语料
[语料百度网盘](https://pan.baidu.com/s/1947bj7WGfz75vZxP22nUZA)

### 更多
**本项目采用所有模型只有在第一次使用时才会开始加载，所以第一次加载速度会有些慢**

更多例子请查看[examples](https://github.com/SeanLee97/xmnlp/tree/master/examples)

## Reference:
本项目采用的数据主要有：

* 人民日报语料
* 结巴分词分词数据
* snownlp情感分析语料

本项目受到以下项目的启发
* [jieba](https://github.com/fxsjy/jieba)
* [snownlp](https://github.com/isnowfy/snownlp)

## License
[MIT](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)
