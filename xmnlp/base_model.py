# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import onnxruntime as ort


class BaseModel(metaclass=ABCMeta):

    def __init__(self, model_path: str):
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    @abstractmethod
    def predict(self):
        raise NotImplementedError
