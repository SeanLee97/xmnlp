# -*- coding: utf-8 -*-

""" XMNLP - Spell Checker

Model Tree:

checker
├── corrector
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── detector
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── vocab.tx

"""

import os

import numpy as np
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

import xmnlp
from xmnlp.utils import rematch, topK


MAX_LEN = 512


class DetectorModel:
    def __init__(self, model_dir):
        # load session and graph
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, ['serve'], export_dir=model_dir)

    def predict(self, token_ids, segment_ids):
        # placeholder
        input_token = self.sess.graph.get_tensor_by_name('Input-Token:0')
        input_segment = self.sess.graph.get_tensor_by_name('Input-Segment:0')
        output = self.sess.graph.get_tensor_by_name('labels/Sigmoid:0')

        return self.sess.run([output], feed_dict={input_token: token_ids,
                                                  input_segment: segment_ids})


class CorrectorModel:
    def __init__(self, model_dir):
        # load session and graph
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, ['serve'], export_dir=model_dir)

    def predict(self, token_ids, segment_ids):
        # placeholder
        input_token = self.sess.graph.get_tensor_by_name('Input-Token:0')
        input_segment = self.sess.graph.get_tensor_by_name('Input-Segment:0')
        output = self.sess.graph.get_tensor_by_name('MLM-Activation/truediv:0')

        return self.sess.run([output], feed_dict={input_token: token_ids,
                                                  input_segment: segment_ids})


class CheckerDecoder:
    def __init__(self, model_dir):
        self.detector = DetectorModel(os.path.join(model_dir, 'detector'))
        self.corrector = CorrectorModel(os.path.join(model_dir, 'corrector'))
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'))
        mask_id = self.tokenizer.encode('[MASK]').ids[1:-1]
        assert len(mask_id) == 1
        self.mask_id = mask_id[0]

    def predict(self, text, suggest=False, k=5, max_k=200):
        tokenized = self.tokenizer.encode(text)
        if len(tokenized.tokens) > MAX_LEN:
            raise ValueError('The text is too long (>512) to process')
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        mapping = rematch(tokenized.offsets)
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        probas = self.detector.predict(token_ids, segment_ids)[0][0]
        incorrect_ids = np.where(probas > 0.5)[0]
        token_ids[0, incorrect_ids] = self.mask_id

        if not suggest:
            ret = []
            for i in incorrect_ids:
                ret.append((i - 1, tokenized.tokens[i]))
            return ret

        probas = self.corrector.predict(token_ids, segment_ids)[0][0]
        sorted_probas, sort_indexs = topK(probas, max_k)
        ret = {}
        for i in incorrect_ids:
            if i == 0 or i == len(tokenized.tokens) - 1:
                continue
            current_token = text[mapping[i][0]: mapping[i][-1] + 1]
            current_pinyin = ' '.join(xmnlp.pinyin(current_token))
            cands = []
            for proba, token in zip(sorted_probas[i], self.tokenizer.decode(sort_indexs[i]).split()):
                pinyin = ' '.join(xmnlp.pinyin(token))
                score = 0
                if current_pinyin == pinyin:
                    score = 1
                cands.append((token, proba + score))
            cands.sort(key=lambda x: x[1], reverse=True)
            ret[(i - 1, current_token)] = cands[:k]
        return dict(ret)
