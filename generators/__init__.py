import os
from abc import ABC
from configparser import ConfigParser


class Generator(ABC):
    def __init__(self, config):
        cfg = ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self._config = cfg[config]


class SimpleGenerator(Generator):
    def search(self, label):
        raise NotImplementedError


class ContextGenerator(Generator):
    def search(self, label, context):
        raise NotImplementedError
