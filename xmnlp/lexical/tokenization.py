# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, List, Tuple

from xmnlp.config.path import ROOT
from xmnlp.lexical import deep_tag


re_english = re.compile(r'[a-zA-Z0-9]+')


class Tokenization:
    def __init__(self,
                 user_vocab_path: Optional[str] = None,
                 detect_new_word: bool = True):
        """ Tokenization
        Arguments:
          user_vocab_path: str, 用户自定义字典, 默认 None
          detect_new_word: bool, 是否识别新词, 默认 True
        """
        # dict.big.txt 是系统词库
        vocab_paths = [os.path.join(ROOT, 'dict.big.txt')]
        if user_vocab_path is not None:
            vocab_paths.append(user_vocab_path)
        self.word2tag, self.max_word_length = self.load_vocab(vocab_paths)
        self.new_word_kernel = deep_tag if detect_new_word else None

    def seg(self, doc: str) -> List[str]:
        """ seg document
        Args:
          doc: str, input document
        """
        return [w[0] for w in self.tag(doc)]

    def tag(self, doc: str) -> List[Tuple[str, str]]:
        """ tag doc
        Args:
          doc: str, input doc
        """
        words = []
        word_length = len(doc)
        while word_length > 0:
            N = min(self.max_word_length, word_length)
            word = doc[-N:]
            while N > 0:
                if word in self.word2tag or N == 1:
                    words.append(word)
                    break
                N -= 1
                word = word[-N:]
            doc = doc[:-N]
            word_length -= N

        # 合并英文和数字
        combine_words = []
        tmp = ''
        for w in words[::-1]:
            if w.encode('utf-8').isalpha() or w.isdigit():
                tmp += w
            else:
                if tmp:
                    combine_words.append((tmp, 'eng'))
                if w.strip():
                    combine_words.append((w, self.word2tag.get(w, 'x')))
                tmp = ''
        if tmp:
            combine_words.append((tmp, 'eng'))
        words = combine_words

        if self.new_word_kernel is None:
            return words

        # new words detect
        final_words = []
        N = len(words)
        i, j = 0, 0
        while i < N:
            if len(words[i][0]) > 1:
                final_words.append(words[i])
                i += 1
                continue
            for k in range(i + 1, N):
                j = k
                if len(words[k][0]) > 1:
                    break
            if i + 1 == j:
                final_words.append(words[i])
            elif i + 1 == N:
                final_words.append(words[i])
                break
            else:
                sequence = ''.join([t[0] for t in words[i:j]])
                final_words += self.new_word_kernel(sequence)
            i = j
        return final_words

    def load_vocab(self, fpaths: List[str]) -> dict:
        word2tag = {}
        max_word_length = 0
        for fpath in fpaths:
            with open(fpath, 'r', encoding='utf-8') as reader:
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    arr = line.split()
                    assert len(arr) in [2, 3], 'supported vocabulary formats: 1)word tag 2)word frequency tag.'
                    if len(arr) == 2:
                        word, tag = arr
                    else:
                        word, _, tag = arr
                    word2tag[word] = tag
                    max_word_length = max(max_word_length, len(word))
        return word2tag, max_word_length
