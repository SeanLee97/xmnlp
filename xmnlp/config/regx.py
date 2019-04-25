# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import unicode_literals
import re
import sys

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


zh = re.compile(r'([\u4E00-\u9FA5]+)', flags=re.UNICODE)
zh_eng = re.compile(r'([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)', flags=re.UNICODE)
skip = re.compile(r'(\r\n|\s)', flags=re.UNICODE)
eng = re.compile(r'[a-zA-Z]+', flags=re.UNICODE)
digt = re.compile(r'([0-9]+\.?[0-9]+|[0-9]+)', flags=re.UNICODE)
endigt = re.compile(r'([0-9]+\.?[0-9]+|[0-9]+|[a-zA-Z]+)', flags=re.UNICODE)
