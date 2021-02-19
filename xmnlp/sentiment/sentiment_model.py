# -*- coding: utf-8 -*-

""" XMNLP - Sentiment

Model Tree:

sentiment
├── saved_model.pb
├── variables
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── vocab.txt

"""

import os

import numpy as np
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer


MAX_LEN = 150


class SentimentModel:
    def __init__(self, model_dir):
        # load session and graph
        self.sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(self.sess, ['serve'], export_dir=model_dir)
        self.tokenizer = BertWordPieceTokenizer(os.path.join(model_dir, 'vocab.txt'))
        self.tokenizer.enable_truncation(max_length=MAX_LEN)

    def predict(self, text):
        tokenized = self.tokenizer.encode(text)
        token_ids = tokenized.ids
        segment_ids = tokenized.type_ids
        token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
        # placeholder
        input_token = self.sess.graph.get_tensor_by_name('Input-Token:0')
        input_segment = self.sess.graph.get_tensor_by_name('Input-Segment:0')
        output = self.sess.graph.get_tensor_by_name('label/Softmax:0')

        probas = self.sess.run([output], feed_dict={input_token: token_ids,
                                                    input_segment: segment_ids})
        return tuple(probas[0][0].tolist())
