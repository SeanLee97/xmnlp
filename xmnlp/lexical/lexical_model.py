# -*- coding: utf-8 -*-

# -------------------------------------------#
# author: sean lee                           #
# email: xmlee97@gmail.com                   #
# -------------------------------------------#

""" XMNLP - Lexical

Model Tree:

lexical
├── label2id.json
├── lexical.onnx
├── trans.npy
└── vocab.txt
"""

import os
import json
from typing import List, Tuple

import numpy as np
from tokenizers import BertWordPieceTokenizer

from xmnlp.base_model import BaseModel
from xmnlp.utils import rematch


MAX_LEN = 512


class LexicalModel(BaseModel):
    def predict(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        token_ids = token_ids.astype('float32')
        segment_ids = segment_ids.astype('float32')
        return self.sess.run(['crf/sub_1:0'], {'Input-Token:0': token_ids,
                                               'Input-Segment:0': segment_ids})


class LexicalDecoder:
    def __init__(self, model_dir, starts=None, ends=None):
        self.trans = np.load(os.path.join(model_dir, 'trans.npy'))
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'), lowercase=True)
        self.lexical_model = LexicalModel(os.path.join(model_dir, 'lexical.onnx'))
        with open(os.path.join(model_dir, 'label2id.json'), encoding='utf-8') as reader:
            label2id = json.load(reader)
            self.id2label = {int(v): k for k, v in label2id.items()}
        self.num_labels = len(self.trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """An elegant viterbi decode implementation

        Modified from https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L404
        """
        # 预处理
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = np.arange(self.num_labels).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for i in range(1, len(nodes)):
            M = scores + self.trans + nodes[i].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            paths = np.concatenate([paths[:, idxs], labels], 0)

        # 最优路径
        return paths[:, scores[:, 0].argmax()]

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokenized = self.tokenizer.encode(text)
        if len(tokenized.tokens) > MAX_LEN:
            raise ValueError('The text is too long (>512) to process')
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        mapping = rematch(tokenized.offsets)
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        nodes = self.lexical_model.predict(token_ids, segment_ids)[0][0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], self.id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]
