# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

from __future__ import absolute_import, unicode_literals

import os
import bz2
import sys

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange
    import cPickle as pickle
else:
    import pickle


class Module(object):
    __notsave__ = []
    __onlysave__ = []

    def filelist(self, fpath):
        """get file list from filename"""

        if os.path.isdir(fpath):
            for root, dirs, files in os.walk(fpath):
                if not dirs:
                    for f in files:
                        yield os.sep.join([root, f])
        else:
            yield fpath

    def save(self, fname, iszip=True):
        """save model"""

        d = {}
        for k, v in self.__dict__.items():
            if self.__onlysave__:
                if k not in self.__onlysave__:
                    continue
            elif k in self.__notsave__:
                continue

            if isinstance(v, set):
                d[k] = list(v)
            elif hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v

        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            with open(fname, 'wb') as wf:
                pickle.dump(d, wf, True)
        else:
            f = bz2.BZ2File(fname, 'wb')
            f.write(pickle.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        """load model"""

        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = pickle.load(open(fname, 'rb'))
        else:
            try:
                f = bz2.BZ2File(fname, 'rb')
                d = pickle.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = pickle.loads(f.read())
            f.close()
        for k, v in d.items():
            if isinstance(self.__dict__[k], set):
                self.__dict__[k] = set(v)
            elif hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v
