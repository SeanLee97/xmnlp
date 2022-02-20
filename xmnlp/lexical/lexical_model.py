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
└── vocab.txt
"""

import os
import re
import json
import logging
from typing import List, Tuple, Optional, Union

import numpy as np
from tokenizers import BertWordPieceTokenizer

from xmnlp.base_model import BaseModel
from xmnlp.utils import rematch


MAX_LEN = 512
re_split = re.compile(r'.*?[\n。]+')


class LexicalModel(BaseModel):
    def predict(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        token_ids = token_ids.astype('float32')
        segment_ids = segment_ids.astype('float32')
        return self.sess.run(['crf/cond_1/Merge'], {
            'Input-Token': token_ids,
            'Input-Segment': segment_ids
        })


class Lexical:
    def __init__(self, model_dir: Optional[str] = None):
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'), lowercase=True)
        self.tokenizer.enable_truncation(max_length=MAX_LEN)
        self.lexical_model = LexicalModel(os.path.join(model_dir, 'lexical.onnx'))
        with open(os.path.join(model_dir, 'label2id.json'), encoding='utf-8') as reader:
            label2id = json.load(reader)
        self.id2label = {int(v): k for k, v in label2id.items()}

    def predict_one(self, text: str, base_position: int = 0) -> List[Tuple[str, str, int, int]]:
        tokenized = self.tokenizer.encode(text)
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        mapping = rematch(tokenized.offsets)
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        logits = self.lexical_model.predict(token_ids, segment_ids)[0][0]
        labels = [self.id2label[i] for i in np.argmax(logits, axis=1)]
        res = []
        for s, e, t in self.bio_decode(labels):
            s = mapping[s]
            s = 0 if not s else s[0]
            e = mapping[e]
            e = len(text) - 1 if not e else e[-1]
            res.append((text[s: e + 1], t, base_position + s, base_position + e + 1))

        return res

    def bio_decode(self, labels: List[str]) -> List[Tuple[int, int, str]]:
        entities = []
        start_tag = None
        for i, tag in enumerate(labels):
            tag_capital = tag.split('-')[0]
            tag_name = tag.split('-')[1] if tag != 'O' else ''
            if tag_capital in ['B', 'O']:
                if start_tag is not None:
                    entities.append((start_tag[0], i - 1, start_tag[1]))
                    start_tag = None
                if tag_capital == 'B':
                    start_tag = (i, tag_name)
            elif tag_capital == 'I' and start_tag is not None and start_tag[1] != tag_name:
                entities.append((start_tag[0], i - 1, start_tag[1]))
                start_tag = (i, tag_name)
        if start_tag is not None:
            entities.append((start_tag[0], i, start_tag[1]))

        return entities

    def transform(self, data: List[Tuple[str, str, int, int]], with_position: bool) -> List[
        Union[Tuple[str, str], Tuple[str, str, int, int]]
    ]:
        if with_position:
            return data
        return [(w, t) for w, t, _, _ in data]

    def predict(self, text: str, with_position: bool = False) -> List[
        Union[Tuple[str, str], Tuple[str, str, int, int]]
    ]:
        if len(text) < MAX_LEN:
            return self.transform(self.predict_one(text), with_position)

        logging.warn('xmnlp: 处理的文本过长（>512），可能会得到意料之外的结果')
        sentences = re_split.findall(text)
        if not sentences:
            return self.transform(self.predict_one(text), with_position)

        result = []
        prev, base_position = 0, 0
        for i in range(1, len(sentences)):
            current_text = ''.join(sentences[prev: i])
            next_text = ''.join(sentences[prev: i + 1])
            if len(current_text) <= MAX_LEN and len(next_text) > MAX_LEN:
                result += self.predict_one(current_text, base_position=base_position)
                prev = i
                base_position += len(current_text)
        result += self.predict_one(''.join(sentences[prev:]), base_position=base_position)
        return self.transform(result, with_position)
