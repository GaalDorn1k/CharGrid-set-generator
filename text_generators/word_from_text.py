import re
import random

from .text_generator_base import TextGeneratorBase


class WordFromText(TextGeneratorBase):
    def __init__(self, text_path: str, alphabet: str = '') -> None:
        super().__init__(alphabet)

        with open(text_path, 'r', encoding='utf-8') as tf:
            words = tf.read().split(' ')

        words = [w for w in words if not w.isspace() and len(w) > 0]
        self.words = []

        for word in words:
            parts = re.split(r'\n|\t|\r', word)
            self.words.extend(parts)

    def generate(self) -> str:
        word = random.choice(self.words)
        word = ''.join([char for char in word if char in self.alphabet])

        if not word or word == '.':
            word = self.alphabet[0]

        return word
