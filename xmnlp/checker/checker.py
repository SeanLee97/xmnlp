# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
#--------------------------------------------#

"""MIT License
Copyright (c) 2018 Sean
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import sys
if sys.version_info[0] == 2:
    reload(sys)
    sys.setdefaultencoding('utf8')
    range = xrange
    import cPickle as pickle
else:
    import pickle

import io
import os

from ..utils import safe_input, filelist
from ..config import path as C_PATH

class Checker(object):

    """
    Args:
        max_edit_distance: 
        verbose: 
            - 0  return best guession
            - 1  all guessions of smallest edit distance 
            - 2  all guessions <= max_edit_distance
    """
    def __init__(self, max_edit_distance=2, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        self.dict = {}
        self.longest_length = 0

        self.train(C_PATH.checker['corpus']['checker'])

    def userdict(self, fpath):
        print("loading...")
        self.train(fpath)

    def load_data(self, fpath):
        datas = []
        for fname in filelist(fpath):
            with io.open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    arr = line.split()
                    if len(arr) > 0:
                        yield safe_input(arr[0])

    def train(self, fname):
        print("start to build dictionary...")
        
        for word in self.load_data(fname):
            if word in self.dict:
                self.dict[word] = (self.dict[word][0], self.dict[word][1] + 1)
            else:
                self.dict[word] = ([], 1)
                self.longest_length = max(self.longest_length, len(word))

            if self.dict[word][1] == 1:
                # first show of word in corpus
                deletes = self.get_deletes(word)
                for d in deletes:
                    if d in self.dict:
                        self.dict[d][0].append(word)
                    else:
                        self.dict[d] = ([word], 0)
        print("done !")

    def get_deletes(self, word):
        dels = []
        queue = [word]
        for d in range(self.max_edit_distance):
            tmp = []
            for word in queue:
                if len(word) > 1:
                    for i in range(len(word)):
                        except_char = word[:i] + word[i+1:]
                        if except_char not in dels:
                            dels.append(except_char)
                        if except_char not in tmp:
                            tmp.append(except_char)
            queue = tmp
        return dels

    def edit_distance(self, seq1, seq2):
        """implement of dameraulevenshtein"""
        m = len(seq1)
        n = len(seq2)
        prev = None 
        cnt = list(range(1, n + 1)) + [0]
        for i in range(m):
            pprev, prev, cnt = prev, cnt, [0]*n + [i+1]
            for j in range(n):
                cnt[j] = min(prev[j] + 1, cnt[j-1]+1, prev[j-1]+(seq1[i]!=seq2[j]))
                if (i > 0 and j > 0 and seq1[i] == seq2[j-1] and seq1[i-1] == seq2[j]
                    and seq1[i] != seq2[j]):
                    cnt[j] = min(cnt[j], pprev[j-2]+1)
        return cnt[n-1]

    def best_match(self, word):
        try:
            return self.word_checker(word)[0]
        except:
            return None 

    def word_checker(self, word):
        if (len(word) - self.longest_length) > self.max_edit_distance or len(word.strip()) == 0:
            return None

        min_guess_len = float('inf')
        queue = [word]
        guess_dict = {}
        tmp_dict = {}

        while len(queue) > 0:
            # queue pop
            cnt = queue[0]
            queue = queue[1:]

            if ((self.verbose < 2) and (len(guess_dict) > 0) and 
                ((len(word) - len(cnt)) > min_guess_len)):
                break

            if (cnt in self.dict) and (cnt not in guess_dict):
                if (self.dict[cnt][1] > 0):
                    guess_dict[cnt] = (self.dict[cnt][1], 
                                            len(word) - len(cnt))
                    if ((self.verbose < 2) and (len(word)==len(cnt))):
                        break

                    elif (len(word) - len(cnt)) < min_guess_len:
                        min_guess_len = len(word) - len(cnt)

                for sc_item in self.dict[cnt][0]:
                    if (sc_item not in guess_dict):
                        
                        if len(cnt) == len(word):
                            item_dist = len(sc_item) - len(cnt)

                        item_dist = self.edit_distance(sc_item, word)
                        
                        if ((self.verbose < 2) and (item_dist > min_guess_len)):
                            pass

                        elif item_dist <= self.max_edit_distance:
                            guess_dict[sc_item] = (self.dict[sc_item][1], item_dist)
                            if item_dist < min_guess_len:
                                min_guess_len = item_dist

                        if self.verbose < 2:
                            guess_dict = {k:v for k, v in guess_dict.items() if v[1] <= min_guess_len}
                    
            if ((self.verbose < 2) and ((len(word)-len(cnt)) > min_guess_len)):
                pass

            elif (len(word)-len(cnt)) < self.max_edit_distance and len(cnt)>1:
                for i in range(len(cnt)): # character index        
                    except_char = cnt[:i] + cnt[i+1:]
                    if except_char not in tmp_dict:
                        queue.append(except_char)
                        tmp_dict[except_char] = None 

        guess_list = guess_dict.items()
        ret = sorted(guess_list, key=lambda x: (x[1][0], -x[1][1]))
        
        if self.verbose==0:
            try:
                return ret[0]
            except:
                return None
        else:
            return ret

    def doc_checker(self, doc):
        if len(doc) < 2:
            return [doc]

        cands = []
        cnt = doc[0]
        appear_times = 0
        for ch in doc[1:]:
            cnt += ch
            suggestion = self.best_match(cnt)

            if len(cands) > 0 and suggestion == cands[-1]:
                appear_times += 1
            else:
                appear_times = 0

            if appear_times > 1:
                cnt = ch
                appear_times = 0
                continue

            if suggestion != None:
                cands.append(suggestion)
            else:
                cands.append(ch)

            if len(cnt) > self.longest_length:
                cnt = ch

        cands = list(dict.fromkeys(cands))
        n = len(cands)
        i = 1
        while i < n:
            if cands[i-1] in cands[i]:
                cands.remove(cands[i-1])
                n = len(cands)
                i -= 2
            i += 1    
            if len(cands) == 1:
                break
        return cands