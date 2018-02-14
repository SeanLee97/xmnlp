# xmnlp

轻量级的中文自然语言处理工具

A Lightweight Chinese Natural Language Processing Toolkit.

# 功能

* 中文分词 & 词性标注
  * 支持自定义词典
* 文本纠错
* 文本摘要 & 关键词提取
* 情感分析
* 文本转拼音

# 安装使用

## 支持环境
```
Linux / Mac os (window 未测试过)
python2 / python3
```
## 安装
```
git clone https://github.com/SeanLee97/xmnlp.git
cd /path/to/xmnlp
pip install -r requirements.txt
python setup.py install
```

# 算法
* 中文分词（采用了和结巴分词类似的算法）：
  * 构建DAG图
  * 动态规划查找，综合正反向（正向加权反向输出）求得DAG最大概率路径
  * 使用了SBME语料训练了一套 HMM + Viterbi 模型，解决未登录词问题
* 文本纠错： bi-gram + levenshtein
* 文本摘要： textrank
* 情感分析： naive bayes
 
# 使用文档
 
## 分词

### example 1
```python
from xmnlp import XmNLP

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

xm = XmNLP(doc)
xm.set_userdict('/path/to/userdict.txt')
print(xm.seg())
```

### example 2
```python
import xmnlp as xm

doc = """自然语言处理: 是人工智能和语言学领域的分支学科。
在这此领域中探讨如何处理及运用自然语言；自然语言认知则是指让电脑“懂”人类的语言。 
自然语言生成系统把计算机数据转化为自然语言。自然语言理解系统把自然语言转化为计算机程序更易于处理的形式。"""

print(xm.seg(doc))
```
result: ['自然语言', '处理', ':', '是', '人工智能', '和', '语言学', '领域', '的', '分支', '学科', '。', '在', '这此', '领域', '中', '探讨', '如何', '处理', '及', '运用', '自然语言', '；', '自然语言', '认知', '则', '是', '指让', '电脑', '“', '懂', '”', '人类', '的', '语言', '。', '自然语言', '生成', '系统', '把', '计算机', '数据', '转化', '为', '自然语言', '。', '自然语言', '理解', '系统', '把', '自然语言', '转化', '为', '计算机程序', '更', '易于', '处理', '的', '形式', '。']
