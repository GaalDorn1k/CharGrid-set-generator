import random

from typing import List
from .text_generator_base import TextGeneratorBase


class RandomText(TextGeneratorBase):
    def __init__(self, words_in_row: List[int], max_word_len: int, alphabet: str) -> None:
        super().__init__(alphabet)
        self.words_in_row = words_in_row
        self.max_word_len = max_word_len

    def generate(self) -> str:
        words = []
        for i in range(random.randint(*self.words_in_row)):
            word = ''
            for j in range(random.randint(1, self.max_word_len)):
                word += random.choice(self.alphabet)
            words.append(word)
        text = ' '.join(words)
        return text
