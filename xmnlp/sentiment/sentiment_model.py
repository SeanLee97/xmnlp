# -*- coding: utf-8 -*-

""" XMNLP - Sentiment

Model Tree:

sentiment
├── sentiment.onnx
└── vocab.txt

"""

import os
from typing import Tuple

import numpy as np
from tokenizers import BertWordPieceTokenizer

from xmnlp.base_model import BaseModel


MAX_LEN = 150


class SentimentModel(BaseModel):
    def predict(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        token_ids = token_ids.astype('float32')
        segment_ids = segment_ids.astype('float32')

        return self.sess.run(['label/Softmax:0'], {'Input-Token:0': token_ids,
                                                   'Input-Segment:0': segment_ids})


class Sentiment:
    def __init__(self, model_dir: str):
        # load session and graph
        self.sentiment_model = SentimentModel(os.path.join(model_dir, 'sentiment.onnx'))
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'), lowercase=True)
        self.tokenizer.enable_truncation(max_length=MAX_LEN)

    def predict(self, text: str) -> Tuple[float, float]:
        tokenized = self.tokenizer.encode(text)
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        probas = self.sentiment_model.predict(token_ids, segment_ids)
        return tuple(probas[0][0].tolist())
