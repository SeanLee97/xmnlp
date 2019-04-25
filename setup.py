# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


__version__ = '0.2.0'

LONGDESC = """
============
xmnlp
============
小明NLP — 轻量级中文自然语言处理工具

A Lightweight Chinese Natural Language Processing Toolkit

详细使用文档见： https://github.com/SeanLee97/xmnlp

功能概述
===============
- 中文分词 & 词性标注
    - 支持简体
    - 支持繁體
    - 支持自定义词典
- 中文拼写检查
- 文本摘要 & 关键词提取
- 情感分析
- 文本转拼音
- 获取偏旁部首

环境说明
===============

支持的操作系统:

- linux
- mac
- windows

支持的python版本：

- python2.7
- python3.5+

安装说明
===============

-  pip安装： ``pip install xmnlp`` / ``pip3 install xmnlp``
-  手动安装： 下载 https://pypi.python.org/pypi/xmnlp/ ，解压运行
   python setup.py install
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    from distutils.util import convert_path

    def _find_packages(where='.', exclude=()):
        """Return a list all Python packages found within directory 'where'

        'where' should be supplied as a "cross-platform" (i.e. URL-style)
        path; it will be converted to the appropriate local path syntax.
        'exclude' is a sequence of package names to exclude; '*' can be used
        as a wildcard in the names, such that 'foo.*' will exclude all
        subpackages of 'foo' (but not 'foo' itself).
        """
        out = []
        stack = [(convert_path(where), '')]
        while stack:
            where, prefix = stack.pop(0)
            for name in os.listdir(where):
                fn = os.path.join(where, name)
                if ('.' not in name and os.path.isdir(fn) and
                        os.path.isfile(os.path.join(fn, '__init__.py'))):
                    out.append(prefix+name)
                    stack.append((fn, prefix + name + '.'))
        for pat in list(exclude)+['ez_setup', 'distribute_setup']:
            from fnmatch import fnmatchcase
            out = [item for item in out if not fnmatchcase(item, pat)]

PUBLISH_CMD = 'python setup.py register sdist upload'

if 'publish' in sys.argv:
    status = subprocess.call(PUBLISH_CMD, shell=True)
    sys.exit(status)


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content

setup(
    name='xmnlp',
    version=__version__,
    description='A Lightweight Chinese Natural Language Processing Toolkit',
    long_description=LONGDESC,
    keywords='chinese segmentation,chinese postager,chinese spell check,pinyin,chinese radical',
    author='sean lee',
    author_email='xmlee97@gmail.com',
    license='MIT License',
    platforms=['all'],
    url='https://github.com/SeanLee97/xmnlp',
    packages=find_packages(exclude=('test*', )),
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    install_requires=['numpy>=1.14.2'],
    package_data={'': ['*.md', '*.txt', '*.pickle', '*.pickle.3']},
    include_package_data=True,
)
