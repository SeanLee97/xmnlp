# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

import os
import sys

def native_content(content):
    if sys.version_info[0] == 2:
        content = content.decode('utf-8')
    return content


def safe_input(content):
    content = content.strip()
    return native_content(content)


def filelist(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if not dirs:
                for f in files:
                    yield os.sep.join([root, f])
    else:
        yield path
