import os
from abc import ABC
from configparser import ConfigParser


class Generator(ABC):
    def __init__(self, config):
        cfg = ConfigParser()
        cfg.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self._config = cfg[config]

    @staticmethod
    def _get_short_labels(label, max_tokens=5):
        """
        Utility function to shorten long labels (that are useless for lookups)
        :param label: the label to shorten
        :param max_tokens: max length for the longest short label
        :return:
        """
        tokens = label.split()
        return [" ".join(tokens[:i+1]) for i in reversed(range(min(max_tokens, len(tokens))))]


class SimpleGenerator(Generator):
    def search(self, label):
        raise NotImplementedError


class ContextGenerator(Generator):
    def search(self, label, context):
        raise NotImplementedError
