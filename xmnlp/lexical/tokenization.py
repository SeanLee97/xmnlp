# -*- coding: utf-8 -*-

import os
from typing import Optional, List, Tuple

from xmnlp import config
from xmnlp.lexical import deep_tag


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
        vocab_paths = [os.path.join(config.MODEL_DIR, 'dict.big.txt')]
        if user_vocab_path is not None:
            vocab_paths.append(user_vocab_path)
        self.word2tag, self.max_word_length = self.load_vocab(vocab_paths)
        self.new_word_kernel = deep_tag if detect_new_word else None

    def seg(self, sentence: str) -> List[str]:
        """ seg sentence
        Args:
          sentence: str, input sentence
        """
        return [w[0] for w in self.tag(sentence)]

    def tag(self, sentence: str) -> List[Tuple[str, str]]:
        """ tag sentence
        Args:
          sentence: str, input sentence
        """
        words = []
        word_length = len(sentence)
        while word_length > 0:
            N = min(self.max_word_length, word_length)
            word = sentence[-N:]
            while N > 0:
                if word in self.word2tag or N == 1:
                    words.append(word)
                    break
                N -= 1
                word = word[-N:]
            sentence = sentence[:-N]
            word_length -= N

        words = words[::-1]
        if self.new_word_kernel is None:
            return words

        # new words detect
        final_words = []
        N = len(words)
        i = 0
        while i < N:
            if len(words[i]) > 1:
                final_words.append((words[i], self.word2tag.get(words[i], 'x')))
                i += 1
                continue
            for j in range(i + 1, N):
                if len(words[j]) > 1:
                    break
            if i + 1 == j:
                final_words.append((words[i], self.word2tag.get(words[i], 'x')))
            elif i + 1 == N:
                final_words.append((words[i], self.word2tag.get(words[i], 'x')))
                break
            else:
                sequence = ''.join(words[i:j])
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
