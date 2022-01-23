# -*- coding: utf-8 -*-

""" XMNLP - SentenceVector

Model Tree:

sentence_vector
├── model.onnx
└── vocab.txt

"""

import os
from typing import Union, Tuple, List, Optional

import numpy as np
from sklearn.neighbors import KDTree
from tokenizers import BertWordPieceTokenizer

from xmnlp import config
from xmnlp.base_model import BaseModel


class SentenceVectorModel(BaseModel):
    def predict(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        token_ids = token_ids.astype('float32')
        segment_ids = segment_ids.astype('float32')

        d = {
            'Input-Token': token_ids,
            'Input-Segment': segment_ids
        }
        return self.sess.run([self.sess.get_outputs()[0].name], d)


class SentenceVector:
    def __init__(self,
                 model_dir: Optional[str] = None,
                 genre: str = '通用',
                 max_length: int = 512):
        """
        Args:
          model_dir: Optional[str], model dir, default None
          genre: str, 内容类型，默认通用，目前支持 ['通用', '金融', '国际'] 三种
          max_length: int, 输入文本的最大长度，默认 512
        """
        assert genre in config.ALLOW_SV_GENRES
        self.genre = genre
        # load session and graph
        if model_dir is None:
            if config.MODEL_DIR is None:
                raise ValueError("Error: 模型地址未设置，请根据文档「安装」 -> 「下载模型」指引下载并配置模型。")
            model_dir = os.path.join(config.MODEL_DIR, 'sentence_vector')

        self.sv_model = SentenceVectorModel(os.path.join(model_dir, 'model.onnx'))
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'), lowercase=True)
        self.tokenizer.enable_truncation(max_length=max_length)

    def transform(self, text: str) -> np.ndarray:
        text = f'{self.genre}[SEP]{text}'
        tokenized = self.tokenizer.encode(text)
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        probas = self.sv_model.predict(token_ids, segment_ids)
        return probas[0][0]

    def similarity(self, x: Union[str, np.ndarray], y: Union[str, np.ndarray]) -> float:
        if isinstance(x, str):
            x = self.transform(x)
        if isinstance(y, str):
            y = self.transform(y)
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def most_similar(self,
                     query: str,
                     docs: List[str],
                     k: int = 1,
                     **kwargs) -> List[Tuple[str, float]]:
        """ get most similar docs
        Args:
          query: str, query sentence
          docs: List[str], a bunch of sentences
          k: int, return size
          kwargs: other parameters of KDTree
        """
        # build tree
        vecs = np.concatenate([np.reshape(self.transform(doc), (1, -1)) for doc in docs], axis=0)
        tree = KDTree(vecs, **kwargs)

        # query vec
        query = np.reshape(self.transform(query), (1, -1))
        distances, indexs = tree.query(query, k=k)
        ret = []
        for idx, dist in zip(indexs[0].tolist(), distances[0].tolist()):
            ret.append((docs[idx], dist))
        return ret
