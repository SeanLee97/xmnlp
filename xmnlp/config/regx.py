# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from __future__ import unicode_literals
import re
import sys

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')


special_tags = [
    ("datetime", re.compile(r"(?:\d{4}[年\-\./]\s*\d{1,2}[月\-\./]\d{1,2}\s{0,1}\d{1,2}:\d{1,2}(?:\:\d{1,2})?)|"
                            r"(?:\d{2,4}\s*年\s*(?:\d{1,2}\s*月\s*)?(?:\d{0,2}\s*[日号]\s*)?)")),
    ("email", re.compile(r"[a-zA-Z0-9\-_\.]+@[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_\.]+", re.IGNORECASE)),
    ("url", re.compile(r"(?:http|ftp)s?://"  # http:// or https://
                       r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|" # domain
                       r"localhost|"  # localhost
                       r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or ip
                       r"(?::\d+)?"  # optional port
                       r"(?:/?|[/?]\S+)[a-zA-Z0-9/]+", re.IGNORECASE)),
    ("html", re.compile(r"\<[a-zA-Z0-9/]+?\>", re.IGNORECASE)),
    ("book", re.compile(r"《.+》"))
]
zh = re.compile(r"([\u4E00-\u9FA5]+)", flags=re.UNICODE)
skip = re.compile(r'\r|\n', flags=re.UNICODE)
blank = re.compile(r"\s")
eng_digt = re.compile(r"([0-9]+\.?[0-9]+|[0-9]+|[a-zA-Z]+|\r|\n)", flags=re.UNICODE)
isalpha = re.compile(r"^[a-zA-Z]*$")
