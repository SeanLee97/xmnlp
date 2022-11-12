<p align='center'><img src='docs/xmnlp-logo.png' width=350 /></p>

<p align='center'>xmnlp: 一款开箱即用的开源中文自然语言处理工具包</p>

<p align='center'>XMNLP: An out-of-the-box Chinese Natural Language Processing Toolkit</p>


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


- 中文词法分析 (RoBERTa + CRF finetune)
   - 分词
   - 词性标注
   - 命名体识别
   - 支持自定义字典
- 中文拼写检查 (Detector + Corrector SpellCheck)
- 文本摘要 & 关键词提取 (Textrank)
- 情感分析 (RoBERTa finetune)
- 文本转拼音 (Trie)
- 汉字偏旁部首 (HashMap)
- [句子表征及相似度计算](https://mp.weixin.qq.com/s/DFUXUnlH_5BlxwyQYeB2xw)


<a name="outline"></a>
# Outline

- [一. 安装](#installation)
  - [模型下载](#installation-download)
  - [配置模型](#installation-configure)
- [二. 使用文档](#usage)
  - [默认分词：seg](#usage-seg)
    - [快速分词：fast_seg](#usage-fast_seg)
    - [深度分词：deep_seg](#usage-deep_seg)
  - [词性标注：tag](#usage-tag)
    - [快速词性标注：fast_tag](#usage-fast_tag)
    - [深度词性标注：deep_tag](#usage-deep_tag)
  - [分词&词性标注自定义字典](#usage-userdict)
  - [命名体识别：ner](#usage-ner)
  - [关键词提取：keyword](#usage-keyword)
  - [关键语句提取：keyphrase](#usage-keyphrase)
  - [情感识别：sentiment](#usage-sentiment)
  - [拼音提取：pinyin](#usage-pinyin)
  - [部首提取：radical](#usage-radical)
  - [文本纠错：checker](#usage-checker)
  - [句子表征及相似度计算：sentence_vector](#usage-sentence_vector)
  - [并行处理](#usage-parallel)
- [三. 更多](#more)
  - [贡献者](#more-contribution)
  - [学术引用](#more-citation)
  - [需求定制](#more-business)
  - [交流群](#more-contact)
- [Refrence](#reference)
- [License](#license)


<a name="installation"></a>
## 一. 安装

<br />安装最新版 xmnlp<br />

```bash
pip install -U xmnlp
```

<br />国内用户可以加一下 index-url<br />

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U xmnlp
```

安装完包之后，还需要下载模型权重才可正常使用

<br />

<a name="installation-download"></a>
### 模型下载

<br />请下载 xmnlp 对应版本的模型，如果不清楚 xmnlp 的版本，可以执行`python -c 'import xmnlp; print(xmnlp.__version__)'` 查看版本<br />


| 模型名称 | 适用版本 | 下载地址 |
| --- | --- | --- |
| xmnlp-onnx-models-v5.zip | v0.5.0, v0.5.1, v0.5.2, v0.5.3 | [飞书](https://wao3cag89c.feishu.cn/file/boxcnppW9Vbd9SSoZEnJdP32Dsg) \[IGHI\] \| [百度网盘](https://pan.baidu.com/s/1YBqD-L5spNg0VOPSDPN3iA) \[l9id\] |
| xmnlp-onnx-models-v4.zip | v0.4.0 | [飞书](https://wao3cag89c.feishu.cn/file/boxcnwdZ9PTtCurhkddlsXrIr0c) \[DKLa\] \| [百度网盘](https://pan.baidu.com/s/1qIHDwXJv18AAv0w72FzrjQ) \[j1qi\] |
| xmnlp-onnx-models-v3.zip | v0.3.2, v0.3.3 | [飞书](https://wao3cag89c.feishu.cn/file/boxcnG5OVqqM8kxtQilt5DachE2) \[o4bA\] \| [百度网盘](https://pan.baidu.com/s/1DsIec7W5CEJ8UNInezgm0Q) \[9g7e\] |


<a name="installation-configure"></a>
### 配置模型

下载模型后需要设置模型路径 xmnlp 才可以正常运行。提供两种配置方式

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

<br />中文分词（默认），基于逆向最大匹配来分词，采用 RoBERTa + CRF 来进行新词识别。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 列表，分词后的结果


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.seg(text))
['xmnlp', '是', '一款', '开箱', '即用', '的', '轻量级', '中文', '自然语言', '处理', '工具', '🔧', '。']
```

<br />

<a name="usage-fast_seg"></a>
### xmnlp.fast_seg(text: str) -> List[str]

<br />基于逆向最大匹配来分词，不包含新词识别，速度较快。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 列表，分词后的结果


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.seg(text))
['xmnlp', '是', '一款', '开箱', '即', '用', '的', '轻量级', '中文', '自然语言', '处理', '工具', '🔧', '。']
```

<br />


<a name="usage-deep_seg"></a>
### xmnlp.deep_seg(text: str) -> List[str]

<br />基于 RoBERTa + CRF 模型，速度较慢。当前深度接口只支持简体中文，不支持繁体。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 列表，分词后的结果


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.deep_seg(text))
['xmnlp', '是', '一款', '开箱', '即用', '的', '轻', '量级', '中文', '自然', '语言', '处理', '工具', '🔧', '。']
```

<br />


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
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.tag(text))
[('xmnlp', 'eng'), ('是', 'v'), ('一款', 'm'), ('开箱', 'n'), ('即用', 'v'), ('的', 'u'), ('轻量级', 'b'), ('中文', 'nz'), ('自然语言', 'l'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'x'), ('。', 'x')]
```

<br />


<a name="usage-fast_tag"></a>
### xmnlp.fast_tag(text: str) -> List[Tuple(str, str)]

<br />基于逆向最大匹配，不包含新词识别，速度较快。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 词和词性元组组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.fast_tag(text))
[('xmnlp', 'eng'), ('是', 'v'), ('一款', 'm'), ('开箱', 'n'), ('即', 'v'), ('用', 'p'), ('的', 'uj'), ('轻量级', 'b'), ('中文', 'nz'), ('自然语言', 'l'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'x'), ('。', 'x')]
```

<br />


<a name="usage-deep_tag"></a>
### xmnlp.deep_tag(text: str) -> List[Tuple(str, str)]

<br />基于 RoBERTa + CRF 模型，速度较慢。当前深度接口只支持简体中文，不支持繁体。<br />
<br />**参数：**<br />

- text：输入文本


<br />**结果返回：**<br />

- 词和词性元组组成的列表


<br />**示例：**<br />

```python
>>> import xmnlp
>>> text = """xmnlp 是一款开箱即用的轻量级中文自然语言处理工具🔧。"""
>>> print(xmnlp.deep_tag(text))
[('xmnlp', 'x'), ('是', 'v'), ('一款', 'm'), ('开箱', 'v'), ('即用', 'v'), ('的', 'u'), ('轻', 'nz'), ('量级', 'b'), ('中文', 'nz'), ('自然', 'n'), ('语言', 'n'), ('处理', 'v'), ('工具', 'n'), ('🔧', 'w'), ('。', 'w')]
```

<br />


<a name="usage-userdict"></a>
### 分词&词性标注自定义字典

支持用户自定义字典，字典格式为

```
词1 词性1
词2 词性2
```

也兼容 jieba 分词的字典格式

```
词1 词频1 词性1
词2 词频2 词性2
```

注：上述行内的间隔符为空格


<br />**使用示例：**<br />

```python
from xmnlp.lexical.tokenization import Tokenization

# 定义 tokenizer
# detect_new_word 定义是否识别新词，默认 True， 设为 False 时速度会更快
tokenizer = Tokenization(user_dict_path, detect_new_word=True)

# 分词
tokenizer.seg(texts)
# 词性标注
tokenizer.tag(texts)
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

<br />


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

<br />

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


<br />


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


<br />


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


<br />


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

<br />


<a name="usage-parallel"></a>
### 并行处理

新版本不再提供对应的并行处理接口，需要使用 `xmnlp.utils.parallel_handler` 来定义并行处理接口。

接口如下：

```python
xmnlp.utils.parallel_handler(callback: Callable, texts: List[str], n_jobs: int = 2, **kwargs) -> Generator[List[Any], None, None]
```

使用示例：

```python
from functools import partial

import xmnlp
from xmnlp.utils import parallel_handler


seg_parallel = partial(parallel_handler, xmnlp.seg)
print(seg_parallel(texts))
```

<br />


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

<p style='font-size: 14px; color: #666666'>
大部分模型基于 <a href='https://github.com/4AI/langml'>LangML</a> 搭建
</p>
