<p align='center'>/ xmnlp /</p>
<p align='center'>小明 NLP — 轻量级中文自然语言处理工具</p>
<p align='center'> A Lightweight Chinese Natural Language Processing Toolkit</p>
<p align='center'>v 0.2.1</p>

<p align='center'><strong>R.I.P</strong> 0.2.1版是 xmnlp 最后一个兼容 Python 2.7 的版本</div>

[![pypi](https://img.shields.io/badge/pypi-v0.2.1-blue.svg)](https://pypi.org/project/xmnlp/)
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

## 使用文档
### 中文分词
```
xmnlp.seg(text[, hmm=True])

- text: 输入文本
- hmm: 是否使用 hmm 算法识别新词
```

```python
import xmnlp

doc = """小明 NLP 是一款开源的轻量级中文自然语言处理工具🔧，当前版本发布时间为2019年9月，改版本修复了一些 bug 也增加了一些特性，主要新增特性如下：分词/词性标注支持日期、email、url、html标签、书📖的识别。如果您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""

xmnlp.seg(doc, hmm=True)
```

分词结果

```
['小明', 'NLP', '是', '一款', '开源', '的', '轻量级', '中文', '自然语言', '处理', '工具', '🔧', '，', '当前', '版本', '发布', '时间', '为', '2019年9月', '，', '改', '版本', '修复', '了', '一些', 'bug', '也', '增加', '了', '一些', '特性', '，', '主要', '新增', '特性', '如下', '：', '分词', '/', '词性', '标注', '支持', '日期', '、', 'email', '、', 'url', '、', 'html', '标签', '、', '书📖', '的', '识别', '。', '如果', '您', '有', '什么', '建议', '/', '疑问', '欢迎', '联系', '我', 'xmlee97@gmail.com']
```

#### 并行分词
```
xmnlp.seg_parallel(texts[, hmm=True, n_jobs=4])

- texts: list of str
- hmm: 是否使用 hmm 算法识别新词
```


```python
import xmnlp

xmnlp.seg_parallel(['结婚的和尚未结婚的都成了和尚',
                    '我喜欢《瓦尔登湖》这本书，如果你也喜欢欢迎联系我 xxx@gmail.com',
                    '<h1>谷歌</h1>的网址是https://google.com',
                    '现在时间是2019年10月'])
```

并行分词结果

```
[['结婚', '的', '和', '尚未', '结婚', '的', '都', '成', '了', '和尚'],
 ['我', '喜欢', '《瓦尔登湖》', '这', '本书', '，', '如果', '你', '也', '喜欢', '欢迎', '联系', '我', 'xxx@gmail.com'],
 ['<h1>', '谷', '歌', '</h1>', '的', '网址', '是', 'https://google.com'],
 ['现在', '时间', '是', '2019年10月']]
```

### 词性标注
```
xmnlp.tag(text[, hmm=True])

- text: 输入文本
- hmm: 是否使用 hmm 算法识别新词
```

```python
import xmnlp

doc = """小明 NLP 是一款开源的轻量级中文自然语言处理工具🔧，当前版本发布时间为2019年9月，改版本修复了一些 bug 也增加了一些特性，主要新增特性如下：分词/词性标注支持日期、email、url、html标签、书📖的识别。如果您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""

xmnlp.tag(doc)
```

词性标注结果

```
[('小明', 'nr'), ('NLP', 'eng'), ('是', 'v'), ('一款', 'm'), ('开源', 'n'), ('的', 'uj'), ('轻量级', 'b'), ('中文', 'nz'), ('自然语言', 'l'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'y'), ('，', 'w'), ('当前', 't'), ('版本', 'n'), ('发布', 'v'), ('时间', 'n'), ('为', 'p'), ('2019年9月', 'datetime'), ('，', 'w'), ('改', 'v'), ('版本', 'n'), ('修复', 'v'), ('了', 'ul'), ('一些', 'm'), ('bug', 'eng'), ('也', 'd'), ('增加', 'v'), ('了', 'ul'), ('一些', 'm'), ('特性', 'n'), ('，', 'w'), ('主要', 'b'), ('新增', 'v'), ('特性', 'n'), ('如下', 't'), ('：', 'w'), ('分词', 'n'), ('/', 'z'), ('词性', 'n'), ('标注', 'v'), ('支持', 'v'), ('日期', 't'), ('、', 'w'), ('email', 'eng'), ('、', 'w'), ('url', 'eng'), ('、', 'w'), ('html', 'eng'), ('标签', 'n'), ('、', 'w'), ('书📖', 'z'), ('的', 'uj'), ('识别', 'v'), ('。', 'w'), ('如果', 'c'), ('您', 'zg'), ('有', 'v'), ('什么', 'r'), ('建议', 'n'), ('/', 'b'), ('疑问', 'v'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xmlee97@gmail.com', 'email')]
```

#### 并行词性标注
```
xmnlp.tag_parallel(texts[, hmm=True, n_jobs=4])

- texts: list of str
- hmm: 是否使用 hmm 算法识别新词
```

```python
import xmnlp

xmnlp.tag_parallel(['结婚的和尚未结婚的都成了和尚',
                    '我喜欢《瓦尔登湖》这本书，如果你也喜欢欢迎联系我 xxx@gmail.com',
                    '<h1>谷歌</h1>的网址是https://google.com',
                    '现在时间是2019年10月'])
```

并行词性标注结果

```
[[('结婚', 'v'), ('的', 'uj'), ('和', 'c'), ('尚未', 'd'), ('结婚', 'v'), ('的', 'uj'), ('都', 'd'), ('成', 'n'), ('了', 'ul'), ('和尚', 'nr')],
 [('我', 'r'), ('喜欢', 'v'), ('《瓦尔登湖》', 'book'), ('这', 'r'), ('本书', 'r'), ('，', 'w'), ('如果', 'c'), ('你', 'r'), ('也', 'd'), ('喜欢', 'v'), ('欢迎', 'v'), ('联系', 'n'), ('我', 'r'), ('xxx@gmail.com', 'email')],
 [('<h1>', 'html'), ('谷', 'nr'), ('歌', 'n'), ('</h1>', 'html'), ('的', 'uj'), ('网址', 'n'), ('是', 'v'), ('https://google.com', 'url')],
 [('现在', 't'), ('时间', 'n'), ('是', 'v'), ('2019年10月', 'datetime')]]
```

#### 用户自定义字典
xmnlp 支持用户自定义字典，只需调用 `xmnlp.set_userdict(/path/to/userdict)` 即可设置自定义字典，自定义字典的格式为：
```
词 词频
词 词频 词性
```
例如
```
自然语言处理 1000 nw
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
xmnlp.set_stopword('/path/to/stopword.txt') # 添加用户自定义停用词

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

# keyword
print(xmnlp.keyword(doc))
# keyphrase
print(xmnlp.keyphrase(doc))
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
xmnlp.radical('自然语言处理')
```
结果输出
```
['自', '灬', '讠', '言', '夂', '王']
```

### 自定义模型
支持用户使用自己的语料训练模型，训练例子在[examples](https://github.com/SeanLee97/xmnlp/tree/master/examples) 的trainer_\*中

### 训练语料
[语料百度网盘](https://pan.baidu.com/s/1947bj7WGfz75vZxP22nUZA)

## 更多
### 关于贡献者
当前本 project 主要者贡献这来自 @4AI 成员，我们期待更多小伙伴的 contributions，一起打造一款简单易用的中文 NLP 工具

### 交流群
欢迎热爱中文 NLP 技术的伙伴加入

- telegram：
- qq 群：

### 学术引用 citation
```
@misc{
  xmnlp,
  title={A Lightweight Chinese Natural Language Processing Toolkit},
  author={Xianming Li},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/SeanLee97/xmnlp}},
}
```
## Reference:
本项目采用的数据主要有：

* 人民日报语料
* 结巴分词分词数据
* snownlp情感分析语料 + 部分作者爬取的语料

本项目受到以下项目的启发

* [jieba](https://github.com/fxsjy/jieba)
* [snownlp](https://github.com/isnowfy/snownlp)

## License
[MIT](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)
