# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#


import os
from setuptools import setup, find_packages


__version__ = '0.3.0'


long_description = open('README.md', encoding='utf-8').read()

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [l for l in f.read().splitlines() if l]

with open('dev-requirements.txt', encoding='utf-8') as f:
    test_requirements = [l for l in f.read().splitlines() if l][1:]


setup(
    name='xmnlp',
    version=__version__,
    description='A Lightweight Chinese Natural Language Processing Toolkit',
    long_description_content_type="text/markdown",
    long_description=long_description,
    keywords='chinese segmentation,chinese postager,chinese spell check,pinyin,chinese radical',
    author='sean lee',
    author_email='xmlee97@gmail.com',
    license='Apache 2.0 License',
    platforms=['all'],
    url='https://github.com/SeanLee97/xmnlp',
    packages=find_packages(exclude=('test*', )),
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    install_requires=requirements,
    tests_require=test_requirements,
    package_data={'': ['*.md', '*.txt', '*.pickle']},
    include_package_data=True,
)
