# !/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

from __future__ import absolute_import, unicode_literals
import io
import sys
from math import log
from xmnlp.module import Module
from xmnlp.utils import safe_input
from .hmm import HMM

if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange


class DAG(Module):
    __notsave__ = []
    __onlysave__ = ['dict']

    def __init__(self, *args):
        self.hmm = True
        self.dict = {}
        self.hmm_segger = None
        self.hmm_tagger = None

    def set_hmm(self, hmm=True):
        self.hmm = hmm

    def train(self, fname):
        counter, total, word_tag = self.load_dict(fname)
        self.dict = {
            'counter': counter,
            'total': total,
            'word_tag': word_tag
        }

    def load_hmm(self, seg_fname=None, tag_fname=None):
        if seg_fname is None and tag_fname is None:
            self.hmm = False
            return

        if seg_fname is not None:
            try:
                self.hmm_segger = HMM(bems=True)
                self.hmm_segger.load(seg_fname)
            except Exception as e:
                print('Error: load seg hmm model failed, ', e)
                self.hmm = False

        if tag_fname is not None:
            try:
                self.hmm_tagger = HMM(bems=False)
                self.hmm_tagger.load(tag_fname)
            except Exception as e:
                print('Error: load tag hmm model failed, ', e)
                self.hmm = False

    def get_freq(self, word):
        total = float(self.dict['total'])
        freq = 1
        for w in self.seg(word):
            freq *= self.dict['counter'].get(w, 1) / total
        freq = max(int(freq * total), self.dict['counter'].get(word, 1))
        return freq

    def userdict(self, fpath, authfreq=False, defaultfreq=5):
        counter, total, word_tag = self.load_dict(fpath, authfreq=authfreq, defaultfreq=defaultfreq)
        self.dict = {
            'counter': dict(self.dict['counter'], **counter),
            'total': self.dict['total'] + total,
            'word_tag': dict(self.dict['word_tag'], **word_tag)
        }

    def load_dict(self, fpath, authfreq=False, defaultfreq=5):
        counter = {}
        total = 0
        word_tag = {}
        for fname in self.filelist(fpath):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, 1):
                    try:
                        line = safe_input(line)

                        arr = line.split()
                        if len(arr) >= 3:
                            word, freq, tag = arr[:3]
                        elif len(arr) == 2:
                            word, x = arr[:2]
                            if x.isdigit():
                                freq = x
                                tag = 'un'
                            else:
                                if authfreq:
                                    freq = self.get_freq(word)
                                else:
                                    freq = defaultfreq
                                tag = x

                        elif len(arr) == 1:
                            word = arr[0]
                            if authfreq:
                                freq = self.get_freq(word)
                            else:
                                freq = defaultfreq
                            tag = 'un'
                        else:
                            continue

                        freq = int(freq)
                        counter[word] = freq
                        total += freq
                        word_tag[word] = tag

                        for ch in range(len(word)):
                            wfrag = word[:ch + 1]
                            if wfrag not in counter:
                                counter[wfrag] = 0
                    except ValueError:
                        raise ValueError('''invalid dictionary entry 
                                            in %s at Line %s: %s''' % (fname, idx, line))

        return counter, total, word_tag

    def get_dag(self, sent):
        dag = {}
        N = len(sent)

        for k in range(N):
            tmp = []
            i = k
            s = sent[k]
            while i < N and s in self.dict['counter']:
                if self.dict['counter'][s]:
                    tmp.append(i)
                i += 1
                s = sent[k: i+1]
            if not tmp:
                tmp.append(k)
            dag[k] = tmp
        return dag

    def get_route(self, sent, dag):
        route = {}
        N = len(sent)
        route[N] = (1, 0)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.dict['counter'].get(sent[idx: x+1]) or 1) - log(self.dict['total']) + route[x+1][0], x) for x in dag[idx])
        return route

    def tag(self, sent):
        s = self.seg(sent)
        lst = list(s)
        ret = self.hmm_tagger.tag(lst)
        for w, f in ret:
            yield w, self.dict['word_tag'].get(w, f)

    def seg(self, sent):
        return self.decode(sent)

    def decode(self, sent):
        dag = self.get_dag(sent)
        route = self.get_route(sent, dag)

        x = 0
        buffer = ''
        N = len(sent)
        while x < N:
            y = route[x][1] + 1
            l_word = sent[x:y]
            if y - x == 1:
                buffer += l_word
            else:
                if buffer:
                    if len(buffer) == 1:
                        yield buffer
                        buffer = ''
                    else:
                        if buffer not in self.dict['counter']:
                            if self.hmm:
                                for item in self.hmm_seg(buffer):
                                    yield item
                            else:
                                for item in buffer:
                                    yield item
                        else:
                            for item in buffer:
                                yield item
                        buffer = ''
                yield l_word
            x = y
        if buffer:
            if len(buffer) == 1:
                yield buffer
            else:
                if buffer not in self.dict['counter']:
                    if self.hmm:
                        for item in self.hmm_seg(buffer):
                            yield item
                    else:
                        for item in buffer:
                            yield item
                else:
                    for item in buffer:
                        yield item

    def hmm_seg(self, sent):
        ret = self.hmm_segger.tag(sent)
        tmp = ''
        for i in ret:
            if i[1] == 'e':
                yield tmp+i[0]
                tmp = ''
            elif i[1] == 'b' or i[1] == 's':
                if tmp:
                    yield tmp
                tmp = i[0]
            else:
                tmp += i[0]
        if tmp:
            yield tmp
