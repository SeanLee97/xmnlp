
<p align='center'>/ xmnlp / </p>

<p align='center'>轻量级中文自然语言处理工具</p>

<p align='center'>A Lightweight Chinese Natural Language Processing Toolkit</p>


<div align='center'>

[![pypi](https://img.shields.io/pypi/v/xmnlp?style=for-the-badge)](https://pypi.org/project/xmnlp/)
[![pypi downloads](https://img.shields.io/pypi/dm/xmnlp?style=for-the-badge)](https://pypi.org/project/xmnlp/)
[![python version](https://img.shields.io/badge/python-3.6,3.7,3.8-orange.svg?style=for-the-badge)]()
[![onnx](https://img.shields.io/badge/onnx,onnxruntime-orange.svg?style=for-the-badge)]()
[![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg?style=for-the-badge)]()
[![GitHub license](https://img.shields.io/github/license/SeanLee97/xmnlp?style=for-the-badge)](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)

</div>


---


<a name="overview"></a>
# 功能概览


- 中文简体词法分析 (RoBERTa + CRF finetune)
   - 分词
   - 词性标注
   - 命名体识别
- 中文拼写检查 (Detector + Corrector SpellCheck)
- 文本摘要 & 关键词提取 (Textrank)
- 情感分析 (RoBERTa finetune)
- 文本转拼音 (Trie)
- 汉字偏旁部首 (HashMap)
- 句子表征及相似度计算


<a name="outline"></a>
# Outline

- [一. 安装](#installation)
  - [安装库](#installation-library)
  - [下载模型](#installation-download)
- [二. 使用文档](#usage)
  - [分词：seg](#usage-seg)
  - [并行分词：seg_parallel](#usage-seg_parallel)
  - [词性标注：tag](#usage-tag)
  - [并行词性标注：tag_parallel](#usage-tag_parallel)
  - [命名体识别：ner](#usage-ner)
  - [并行命名体识别：ner_parallel](#usage-ner_parallel)
  - [关键词提取：keyword](#usage-keyword)
  - [并行关键词提取：keyword_parallel](#usage-keyword_parallel)
  - [关键语句提取：keyphrase](#usage-keyphrase)
  - [并行关键语句提取：keyphrase_parallel](#usage-keyphrase_parallel)
  - [情感识别：sentiment](#usage-sentiment)
  - [并行情感识别：sentiment_parallel](#usage-sentiment_parallel)
  - [拼音提取：pinyin](#usage-pinyin)
  - [并行拼音提取：pinyin_parallel](#usage-pinyin_parallel)
  - [部首提取：radical](#usage-radical)
  - [并行部首提取：radical_parallel](#usage-radical_parallel)
  - [文本纠错：checker](#usage-checker)
  - [并行文本纠错：checker_parallel](#usage-checker_parallel)
  - [句子表征及相似度计算：sentence_vector](#usage-sentence_vector)
- [三. 更多](#more)
  - [贡献者](#more-contribution)
  - [学术引用](#more-citation)
  - [需求定制](#more-business)
  - [交流群](#more-contact)
- [Refrence](#reference)
- [License](#license)


<a name="installation"></a>
## 一. 安装


<a name="installation-library"></a>
### 1. 安装库


<a name="installation-library-pip"></a>
#### 方式 1

<br />安装最新版 xmnlp<br />

```bash
pip install -U xmnlp
```

<br />国内用户可以加一下 index-url<br />

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U xmnlp
```


<a name="installation-library-manually"></a>
#### 方式 2


```bash
git clone https://github.com/SeanLee97/xmnlp.git
cd /path/to/xmnlp
pip install -r requirements.txt
python setup.py install
```


<a name="installation-download"></a>
### 2. 下载模型


<a name="installation-download-url"></a>
#### 下载地址

<br />请下载 xmnlp 对应版本的模型，如果不清楚 xmnlp 的版本，可以执行`python -c 'import xmnlp; print(xmnlp.__version__)'` 查看版本<br />


| 模型名称 | 适用版本 | 下载地址 |
| --- | --- | --- |
| xmnlp-onnx-models-v4.zip | v0.4.0 | [飞书](https://wao3cag89c.feishu.cn/file/boxcnwdZ9PTtCurhkddlsXrIr0c) \[DKLa\] \| [百度网盘](https://pan.baidu.com/s/1qIHDwXJv18AAv0w72FzrjQ) \[j1qi\] |
| xmnlp-onnx-models-v3.zip | v0.3.2, v0.3.3 | [飞书](https://wao3cag89c.feishu.cn/file/boxcnG5OVqqM8kxtQilt5DachE2) \[o4bA\] \| [百度网盘](https://pan.baidu.com/s/1DsIec7W5CEJ8UNInezgm0Q) \[9g7e\] |

\* 模型已切换至 onnx， 先前 tensorflow 模型已不可用，请下载最新版本模型

\* 如果下载地址失效了，烦请提 issue 反馈<br />

<a name="installation-download-setting"></a>
#### 模型设置

提供两种设置方式

**方式 1：配置环境变量（推荐）**

<br />下载好的模型解压后，可以设置环境变量指定模型地址。以 Linux 系统为例，设置如下<br />

```bash
export XMNLP_MODEL=/path/to/xmnlp-models
```


**方式 2：通过函数设置**

<br />在调用 xmnlp 前设置模型地址，如下<br />

```python
import xmnlp

xmnlp.set_model('/path/to/xmnlp-models')
```

<br />* 上述 `/path/to/` 只是占位用的，配置时请替换成模型真实的目录地址。<br />

<a name="usage"></a>
## 二. 使用文档


<a name="usage-seg"></a>
### xmnlp.seg(text: str) -> List[str]

<br />中文分词。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 列表，分词后的结果


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开源的轻量级中文自然语言处理工具🔧。
   ...: 功能涵盖：中文分词、词性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。
   ...: 如果您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""
>>> print(xmnlp.seg(text))
['xmnlp', '是', '一款', '开源', '的', '轻量级', '中文', '自然语言', '处理', '工具', '🔧', '。', '功能', '涵盖', '：', '中文', '分', '词', '、', '词性', '标注', '、', '命名体识别', '、', '文本', '纠错', '，', '情感识别', '，', '文本', '转', '拼音', '、', '提取', '文', '本', '部', '首等', '。', '如果', '您', '有', '什么', '建议', '/', '疑问', '欢迎', '联系', '我', 'xmlee97', '@', 'gmail', '.', 'com']
```


<a name="usage-seg_parallel"></a>
### xmnlp.seg_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[str], None, None]:

<br />并行处理中文分词。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ['xmnlp 是一款开源的轻量级中文自然语言处理工具🔧',
   ...:      '功能涵盖：中文分词、词性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。']
>>> print(list(xmnlp.seg_parallel(texts)))
[['xmnlp', '是', '一款', '开源', '的', '轻量级', '中文', '自然语言', '处理', '工具', '🔧'], ['功能', '涵盖', '：', '中文', '分词', '、', '词', '性', '标注', '、', '命名体识别', '、', '文本', '纠错', '，', '情感识别', '，', '文本', '转', '拼音', '、', '提取', '文本', '部', '首等', '。']]
```


<a name="usage-tag"></a>
### xmnlp.tag(text: str) -> List[Tuple(str, str)]

<br />词性标注。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 词和词性元组组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开源的轻量级中文自然语言处理工具🔧。
   ...: 功能涵盖：中文分词、词性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。
   ...: 如果您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""
>>> print(xmnlp.tag(text))
[('xmnlp', 'x'), ('是', 'v'), ('一款', 'm'), ('开源', 'v'), ('的', 'u'), ('轻量级', 'b'), ('中文', 'nz'), ('自然语言', 'nz'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'n'), ('。', 'w'), ('功能', 'n'), ('涵盖', 'v'), ('：', 'w'), ('中文', 'nz'), ('分', 'q'), ('词', 'n'), ('、', 'w'), ('词性', 'nz'), ('标注', 'v'), ('、', 'w'), ('命名体识别', 'g'), ('、', 'w'), ('文本', 'n'), ('纠错', 'vn'), ('，', 'w'), ('情感识别', 'nz'), ('，', 'w'), ('文本', 'n'), ('转', 'v'), ('拼音', 'n'), ('、', 'w'), ('提取', 'v'), ('文', 'ng'), ('本', 'r'), ('部', 'q'), ('首等', 'b'), ('。', 'w'), ('如果', 'c'), ('您', 'r'), ('有', 'v'), ('什么', 'r'), ('建议', 'n'), ('/', 'w'), ('疑问', 'n'), ('欢迎', 'v'), ('联系', 'vn'), ('我', 'r'), ('xmlee97', 'x'), ('@', 'w'), ('gmail', 'x'), ('.', 'w'), ('com', 'x')]
```


<a name="usage-tag_parallel"></a>
### xmnlp.tag_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[Tuple(str, str)], None, None]:

<br />并行处理词性标注。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ['xmnlp 是一款开源的轻量级中文自然语言处理工具🔧',
   ...:      '功能涵盖：中文分词、词性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。']
>>> print(list(xmnlp.tag_parallel(texts)))
[[('xmnlp', 'x'), ('是', 'v'), ('一款', 'm'), ('开源', 'v'), ('的', 'u'), ('轻量级', 'b'), ('中文', 'nz'), ('自然语言', 'nz'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'w')], [('功能', 'n'), ('涵盖', 'v'), ('：', 'w'), ('中文', 'nz'), ('分词', 'n'), ('、', 'w'), ('词', 'n'), ('性', 'ng'), ('标注', 'v'), ('、', 'w'), ('命名体识别', 'nz'), ('、', 'w'), ('文本', 'n'), ('纠错', 'vn'), ('，', 'w'), ('情感识别', 'nz'), ('，', 'w'), ('文本', 'n'), ('转', 'v'), ('拼音', 'n'), ('、', 'w'), ('提取', 'v'), ('文本', 'n'), ('部', 'q'), ('首等', 'v'), ('。', 'w')]]
```

<br />

<a name="usage-ner"></a>
### xmnlp.ner(text: str) -> List[Tuple(str, str, int, int)]

<br />命名体识别，支持识别的实体类型为：

- TIME：时间
- LOCATION：地点
- PERSON：人物
- JOB：职业
- ORGANIZAIRION：机构


<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 实体、实体类型、实体起始位置和实体结尾位置组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = "现任美国总统是拜登。"
>>> print(xmnlp.ner(text))
[('美国', 'LOCATION', 2, 4), ('总统', 'JOB', 4, 6), ('拜登', 'PERSON', 7, 9)]
```


<a name="usage-ner_parallel"></a>
### xmnlp.ner_parallel(texts: List[str], n_jobs: int = 2) -> Generator[List[Tuple(str, str, int, int)] None, None]:

<br />并行处理命名体识别。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ['现任美国总统是拜登',
   ...:      '前任美国总统是特朗普']
>>> print(list(xmnlp.tag_parallel(texts)))
[[('美国', 'LOCATION', 2, 4), ('总统', 'JOB', 4, 6), ('拜登', 'PERSON', 7, 9)], [('美国', 'LOCATION', 2, 4), ('总统', 'JOB', 4, 6), ('特朗普', 'PERSON', 7, 10)]]
```


<a name="usage-keyword"></a>
### xmnlp.keyword(text: str, k: int = 10, stopword: bool = True, allowPOS: Optional[List[str]] = None) -> List[Tuple[str, float]]

<br />从文本中提取关键词，基于 Textrank 算法。<br />
<br />**参数：**<br />

- text：文本输入
- k：返回关键词的个数
- stopword：是否去除停用词
- allowPOS：配置允许的词性


<br />**结果返回：**<br />

- 由关键词和权重组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """自然语言处理: 是人工智能和语言学领域的分支学科。
    ...: 在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的
    ...: 语言。
    ...: 自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化
    ...: 为计算机程序更易于处理的形式。"""
>>> print(xmnlp.keyword(text))
[('自然语言', 2.3000579596585897), ('语言', 1.4734141257937314), ('计算机', 1.3747500999598312), ('转化', 1.2687686226652466), ('系统', 1.1171384775870152), ('领域', 1.0970728069617324), ('人类', 1.0192131829490039), ('生成', 1.0075197087342542), ('认知', 0.9327188339671753), ('指', 0.9218423928455112)]
```

<br />

<a name="usage-keyword_parallel"></a>
### xmnlp.keyword_parallel(texts: List[str], k: int = 10, stopword: bool = True, allowPOS: Optional[List[str]] = None, n_jobs: int = 2) -> Generator[List[Tuple[str, float]], None, None]

<br />并行处理关键词提取。<br />
<br />**参数：**<br />

- texts：文本列表
- k：返回关键词的个数
- stopword：是否去除停用词
- allowPOS：配置允许的词性
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> docs = ["""自然语言处理: 是人工智能和语言学领域的分支学科。\n在这此领域中探讨
    ...: 如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 \n自然语言
    ...: 生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机
    ...: 程序更易于处理的形式。""",
    ...:    """xmnlp 是一款开源的轻量级中文自然语言处理工具🔧。  功能涵盖：中文分词、词
    ...: 性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。  如果
    ...: 您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""]
>>> print(list(xmnlp.keyword_parallel(docs)))
[[('自然语言', 2.3000579596585897), ('语言', 1.4734141257937314), ('计算机', 1.3747500999598312), ('转化', 1.2687686226652466), ('系统', 1.1171384775870152), ('领域', 1.0970728069617324), ('人类', 1.0192131829490039), ('生成', 1.0075197087342542), ('认知', 0.9327188339671753), ('指', 0.9218423928455112)], [('文本', 1.4318894662141104), ('中文', 1.3756785780080754), ('建议', 1.1420258441127817), ('提取文本', 1.1214039563905946), ('拼音', 1.106313873517177), ('分词', 1.0886571300192784), ('词', 1.0816698383766425), ('标注', 1.0773997802582003), ('疑问', 1.0279032612606591), ('工具', 1.02290616843714)]]
```


<a name="usage-keyphrase"></a>
### xmnlp.keyphrase(text: str, k: int = 10, stopword: bool = False) -> List[str]

<br />从文本中提取关键句，基于 Textrank 算法。<br />
<br />**参数：**<br />

- text：文本输入
- k：返回关键词的个数
- stopword：是否去除停用词


<br />**结果返回：**<br />

- 由关键词和权重组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """自然语言处理: 是人工智能和语言学领域的分支学科。
    ...: 在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的
    ...: 语言。
    ...: 自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化
    ...: 为计算机程序更易于处理的形式。"""
>>> print(xmnlp.keyphrase(text, k=2))
['自然语言理解系统把自然语言转化为计算机程序更易于处理的形式', '自然语言生成系统把计算机数据转化为自然语言']
```

<br />

<a name="usage-keyphrase_parallel"></a>
### xmnlp.keyphrase_parallel(texts: List[str], k: int = 10, stopword: bool = False, n_jobs: int = 2) -> Generator[List[str], None, None]

<br />并行处理关键句提取。<br />
<br />**参数：**<br />

- texts：文本列表
- k：返回关键词的个数
- stopword：是否去除停用词
- allowPOS：配置允许的词性
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> docs = ["""自然语言处理: 是人工智能和语言学领域的分支学科。\n在这此领域中探讨
    ...: 如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 \n自然语言
    ...: 生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机
    ...: 程序更易于处理的形式。""",
    ...:    """xmnlp 是一款开源的轻量级中文自然语言处理工具🔧。  功能涵盖：中文分词、词
    ...: 性标注、命名体识别、文本纠错，情感识别，文本转拼音、提取文本部首等。  如果
    ...: 您有什么建议/疑问欢迎联系我 xmlee97@gmail.com"""]
>>> print(list(xmnlp.keyphrase_parallel(docs, k=2)))
[['自然语言理解系统把自然语言转化为计算机程序更易于处理的形式', '自然语言生成系统把计算机数据转化为自然语言'], ['如果您有什么建议/疑问欢迎联系我xmlee97@gmail.com', '文本转拼音、提取文本部首等']]
```


<a name="usage-sentiment"></a>
### xmnlp.sentiment(text: str) -> Tuple[float, float]

<br />情感识别，基于电商评论语料训练，适用于电商场景下的情感识别。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 元组，格式为：[负向情感概率，正向情感概率]


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = "这本书真不错，下次还要买"
>>> print(xmnlp.sentiment(text))
(0.02727833203971386, 0.9727216958999634)
```


<a name="usage-sentiment_parallel"></a>
### xmnlp.sentiment_parallel(texts: List[str], n_jobs=2) -> Generator[Tuple[float, float], None, None]

<br />并行处理情感识别。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ["这本书真不错，下次还要买", "垃圾书一本"]
>>> print(list(xmnlp.sentiment_parallel(text)))
[(0.02727833203971386, 0.9727216958999634),
 (0.9930351972579956, 0.006964794360101223)]
```


<a name="usage-pinyin"></a>
### xmnlp.pinyin(text: str) -> List[str]

<br />文本转拼音<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 拼音组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = "自然语言处理"
>>> print(xmnlp.pinyin(text))
['Zi', 'ran', 'yu', 'yan', 'chu', 'li']
```


<a name="usage-pinyin_parallel"></a>
### xmnlp.pinyin_parallel(texts: List[str], n_jobs=2) -> Generator[List[str], None, None]

<br />并行处理文本转拼音。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ["自然语言处理", "人工智能"]
>>> print(list(xmnlp.pinyin_parallel(text)))
[['Zi', 'ran', 'yu', 'yan', 'chu', 'li'], ['ren', 'gong', 'zhi', 'neng']]
```


<a name="usage-radical"></a>
### xmnlp.radical(text: str) -> List[str]

<br />提取文本部首<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 部首组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = "自然语言处理"
>>> print(xmnlp.radical(text))
['自', '灬', '讠', '言', '夂', '王']
```


<a name="usage-radical_parallel"></a>
### xmnlp.radical_parallel(texts: List[str], n_jobs=2) -> Generator[List[str], None, None]

<br />并行处理文本部首提取。<br />
<br />**参数：**<br />

- texts：文本列表
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ["自然语言处理", "人工智能"]
>>> print(list(xmnlp.radical_parallel(text)))
[['自', '灬', '讠', '言', '夂', '王'], ['人', '工', '日', '月']]
```


<a name="usage-checker"></a>
### xmnlp.checker(text: str, suggest: bool = True, k: int = 5, max_k: int = 200) -> Union[ List[Tuple[int, str]], Dict[Tuple[int, str], List[Tuple[str, float]]]]:

<br />文本纠错<br />
<br />**参数：**<br />

- text：输入文本
- suggest：是否返回建议词
- k：返回建议词的个数
- max_k：拼音搜索最大次数（建议保持默认值）


<br />**结果返回：**<br />

- suggest 为 False 时返回 (错词下标，错词) 列表；suggest 为 True 时返回字典，字典键为(错词下标，错词) 列表，值为建议词以及权重列表。


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = "不能适应体育专业选拔人材的要求"
>>> print(xmnlp.checker(text))
{(11, '材'): [('才', 1.58528071641922), ('材', 1.0009655653266236), ('裁', 1.0000178480604518), ('员', 0.35814568400382996), ('士', 0.011077565141022205)]}
```


<a name="usage-checker_parallel"></a>
### xmnlp.checker_parallel(texts: List[str], suggest: bool = True, k: int = 5, max_k: int = 200, n_jobs: int = 2) -> Generator[ Union[List[Tuple[int, str]], Dict[Tuple[int, str], List[Tuple[str, float]]] ], None, None]:

<br />并行处理文本纠错。<br />
<br />**参数：**<br />

- texts：文本列表
- suggest：是否返回建议词
- k：返回建议词的个数
- max_k：拼音搜索最大次数（建议保持默认值）
- n_jobs: 线程 worker 数


<br />**结果返回：**<br />

- generator


<br />**示例：**<br />

```python
>>> import xmnlp
>>> texts = ['开展公共资源交易活动监督检查和举报投拆处理。', '不能适应体育专业选拔人材的要求。', '比对整治前后影相资料。', '保护好堪查现场。']
>>> print(list(xmnlp.checker_parallel(text)))
[{(18, '拆'): [('资', 0.41119733452796936), ('诉', 0.21499130129814148), ('票', 0.11507325619459152), ('入', 0.07330290228128433), ('明', 0.009785536676645279)]}, {(11, '材'): [('才', 1.58528071641922), ('材', 1.0009655653266236), ('裁', 1.0000178480604518), ('员', 0.35814568400382996), ('士', 0.011077565141022205)]}, {(7, '相'): [('响', 1.1048823446035385), ('像', 1.0515491589903831), ('相', 1.000226703719818), ('乡', 1.0002082456485368), ('的', 0.29209405183792114)]}, {(3, '堪'): [('看', 1.0040899985469878), ('勘', 1.00186610117089), ('检', 0.16447395086288452), ('调', 0.1378173977136612), ('考', 0.0857236310839653)]}]
```

<a name="usage-sentence_vector"></a>
### xmnlp.sv.SentenceVector(model_dir: Optional[str] = None, genre: str = '通用', max_length: int = 512)

SentenceVector 初始化函数

- model_dir: 模型保存地址，默认加载 xmnlp 提供的模型权重
- genre: 内容类型，目前支持 ['通用', '金融', '国际'] 三种
- max_length: 输入文本的最大长度，默认 512

以下是 SentenceVector 的三个成员函数

### xmnlp.sv.SentenceVector.transform(self, text: str) -> np.ndarray
### xmnlp.sv.SentenceVector.similarity(self, x: Union[str, np.ndarray], y: Union[str, np.ndarray]) -> float
### xmnlp.sv.SentenceVector.most_similar(self, query: str, docs: List[str], k: int = 1, **kwargs) -> List[Tuple[str, float]]

- query: 查询内容
- docs: 文档列表
- k: 返回 topk 相似文本
- kwargs: KDTree 的参数，详见 [sklearn.neighbors.KDTree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html)

使用示例

```python
import numpy as np
from xmnlp.sv import SentenceVector


query = '我想买手机'
docs = [
    '我想买苹果手机',
    '我喜欢吃苹果'
]

sv = SentenceVector(genre='通用')
for doc in docs:
    print('doc:', doc)
    print('similarity:', sv.similarity(query, doc))
print('most similar doc:', sv.most_similar(query, docs))
print('query representation shape:', sv.transform(query).shape)
```

输出

```
doc: 我想买苹果手机
similarity: 0.68668646
doc: 我喜欢吃苹果
similarity: 0.3020076
most similar doc: [('我想买苹果手机', 16.255546509314417)]
query representation shape: (312,)
```

<a name="more"></a>
## 三. 更多


<a name="more-contribution"></a>
### 关于贡献者

<br />期待更多小伙伴的 contributions，一起打造一款简单易用的中文 NLP 工具 <br />

<a name="more-citation"></a>
### 学术引用 Citation


```python
@misc{
  xmnlp,
  title={XMNLP: A Lightweight Chinese Natural Language Processing Toolkit},
  author={Xianming Li},
  year={2018},
  publisher={GitHub},
  howpublished={\url{https://github.com/SeanLee97/xmnlp}},
}
```

<br />

<a name="more-business"></a>
### 需求定制

<br />本人致力于 NLP 研究和落地，方向包括：信息抽取，情感分类等。<br />
<br />其他 NLP 落地需求可以联系 [xmlee97@gmail.com](mailto:xmlee97@gmail.com) （此为有偿服务，xmnlp 相关的 bug 直接提 issue）<br />
<br />

<a name="more-contact"></a>
### 交流群

<br />搜索公众号 `xmnlp-ai` 关注，菜单选择 “交流群” 入群。<br />
<br />

<a name="reference"></a>
## Reference

<br />本项目采用的数据主要有：<br />

- 词法分析，文本纠错：人民日报语料
- 情感识别：[ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)


<a name="license"></a>
## License

<br />[Apache 2.0](https://github.com/SeanLee97/xmnlp/blob/master/LICENSE)<br />
<br />
