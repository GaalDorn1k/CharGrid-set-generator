import os
import cv2
import random
import numpy as np

from text_generators import *
from typing import Union, Tuple
from PIL import Image, ImageDraw, ImageFont


class RowGenerator():
    def __init__(self, alphabet: str, fonts_path: str, font_size: Union[int, list],
                 text_generator: dict, font_color_range: list = [[0, 255], [0, 255], [0, 255]]) -> None:
        self.alphabet = alphabet
        self.text_gen = globals()[list(text_generator.keys())[0]](**list(text_generator.values())[0], alphabet=alphabet)

        _font_exs = ('ttf', 'TTF')
        if os.path.isdir(fonts_path):
            self.fonts_names = {f: os.path.join(fonts_path, f) for f in os.listdir(fonts_path) if f.endswith(_font_exs)}
        elif fonts_path.endswith(_font_exs):
            self.fonts_names = {fonts_path[fonts_path.rfind('/') + 1:]: fonts_path}
        elif not self.fonts_names:
            raise FileExistsError('font_path must be either a directory containing .ttf fonts or a path to a .ttf font')

        if isinstance(font_size, list):
            self.font_size = font_size
        elif isinstance(font_size, int):
            self.font_size = (font_size, font_size)

        self.font_color_range = font_color_range

    def _union_rects(self, rects: np.ndarray) -> Tuple[int, int, int, int]:
        rects = rects.T
        x = min(rects[0])
        y = min(rects[1])
        w = max(rects[0] + rects[2]) - x
        h = max(rects[1] + rects[3]) - y
        return x, y, w, h

    def _get_char_code(self, char: str) -> int:
        if self.alphabet.find(char) >= 0:
            return self.alphabet.find(char) + 1

    def generate(self, text: str = None, font_name: str = None, font_size: tuple = None,
                 font_color: Tuple[int, int, int] = None, bold: bool = False) -> Tuple[Image.Image, np.ndarray, str]:
        if not text:
            text = self.text_gen.generate()

        if font_name not in self.fonts_names.keys():
            font_name = random.choice(list(self.fonts_names.values()))
        else:
            font_name = self.fonts_names[font_name]

        if not font_size:
            font_size = random.uniform(*self.font_size)

        if not font_color:
            font_color = (random.randint(*self.font_color_range[0]),
                          random.randint(*self.font_color_range[1]),
                          random.randint(*self.font_color_range[2]))

        stroke_width = 0

        if bold:
            stroke_width = 1

        _font_base_size = 50
        font = ImageFont.truetype(font_name, size=_font_base_size)

        text_size = list(font.getsize(text))
        text_size[0] = int(text_size[0] * 1.5)
        text_size[1] = int(text_size[1] * 2)
        text_crop = Image.new(size=text_size, mode='RGBA', color=(255, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_crop)
        grid_crop = np.zeros((len(self.alphabet) + 1, text_size[1], text_size[0]))
        zero_layer = np.ones((text_size[1], text_size[0]))
        step = 0

        for char in text:
            char_size = font.getsize(char)

            if char == ' ':
                step += char_size[0]
                continue

            char_crop = Image.new(size=char_size, mode='RGB', color=(255, 255, 255))
            char_draw = ImageDraw.Draw(char_crop)
            char_draw.text((0, 0), char, font=font, fill=(0, 0, 0),
                           stroke_width=stroke_width, stroke_fill=(0, 0, 0))
            char_crop = np.array(char_crop)
            gray = cv2.cvtColor(char_crop, cv2.COLOR_BGR2GRAY)
            gray = 255 - gray
            char_crop, char_crop = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cnts = cv2.findContours(char_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            rects = []

            for cnt in cnts:
                rects.append(list(cv2.boundingRect(cnt)))

            if len(rects) > 1:
                x, y, w, h = self._union_rects(np.array(rects))

            else:
                x, y, w, h = rects[0]

            char_code = self._get_char_code(char)
            grid_crop[char_code] = cv2.rectangle(grid_crop[char_code],
                                                 (step + x, y),
                                                 (step + x + w, y + h),
                                                 1, -1)
            zero_layer = cv2.rectangle(zero_layer,
                                       (step + x, y),
                                       (step + x + w, y + h),
                                       0, -1)
            text_draw.text((step, 0), char, font=font, fill=font_color,
                           stroke_width=stroke_width, stroke_fill=font_color)
            step += char_size[0]

        ys, xs = np.where(zero_layer == 0)
        box = (min(xs), min(ys), max(xs), int(max(ys) * 1.05))
        text_crop = text_crop.crop(box)
        zero_layer = zero_layer[box[1]:box[3], box[0]:box[2]]
        grid_crop = grid_crop[:, box[1]:box[3], box[0]:box[2]]
        scale = random.choice(self.font_size) / _font_base_size
        y = int(text_crop.size[1] * scale)
        x = int(y * text_crop.size[0] / text_crop.size[1])
        new_size = (x, y)
        text_crop = text_crop.resize(new_size)
        resized_grid_crop = []

        for layer in grid_crop:
            layer = cv2.resize(layer, new_size, interpolation=cv2.INTER_NEAREST)
            resized_grid_crop.append(layer)

        grid_crop = np.array(resized_grid_crop)

        return text_crop, grid_crop, text
