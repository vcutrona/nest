import os
from abc import ABC
from configparser import ConfigParser


class SimpleGenerator:

    def __init__(self):
        self._config = ConfigParser()
        self._config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

    def search(self, label):
        raise NotImplementedError


class ContextGenerator(SimpleGenerator, ABC):
    def search_context(self, label, context):
        raise NotImplementedError
