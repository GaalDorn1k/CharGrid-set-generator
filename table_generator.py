import random
import numpy as np

from collections import namedtuple
from copy import copy
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from row_generator import RowGenerator


class TableGenerator():
    def __init__(self, cells_range: tuple, text_generator: RowGenerator, size_range: tuple) -> None:
        self.text_generator = text_generator
        self.cells_range = cells_range
        self.size_range = size_range

    def generate(self) -> Image.Image:
        tdata = []
        cells = (random.randint(1, self.cells_range[0]),
                 random.randint(1, self.cells_range[1]))

        for y in range(cells[1]):
            row = []
            for x in range(cells[0]):
                text = self.text_generator.text_gen.generate()
                row.append(text)
            tdata.append(row)

        align = [random.choice(['l', 'r', 'c'])] * cells[0]
        width = random.randint(2, 5)
        table, grid = self._plot_table(tdata, align=align, width=width)

        return table, grid, tdata

    def _position_tuple(self, *args) -> tuple:
        Position = namedtuple('Position', ['top', 'right', 'bottom', 'left'])
        if len(args) == 0:
            return Position(0, 0, 0, 0)
        elif len(args) == 1:
            return Position(args[0], args[0], args[0], args[0])
        elif len(args) == 2:
            return Position(args[0], args[1], args[0], args[1])
        elif len(args) == 3:
            return Position(args[0], args[1], args[2], args[1])
        else:
            return Position(args[0], args[1], args[2], args[3])

    def _plot_table(self, table: List[str],
                    cell_pad=(30, 5), margin=(1, 1), align=['l', 'r', 'c'], colors={}, width=2) -> Image.Image:
        """
        Draw a table using only Pillow
        table:    an 2d list, must be str
        cell_pad: padding for cell, (top_bottom, left_right)
        margin:   margin for table, css-like shorthand
        align:    None or list, 'l'/'c'/'r' for left/center/right, length must be the max count of columns
        colors:   dict, as follows
        """

        font = ImageFont.truetype("/home/user0/projects/O/data_for_gen/fonts/a_AlgeriusNr.TTF", 30)
        _color = {
            'bg': (0, 0, 0, 0),
            'cell_bg': (0, 0, 0, 0),
            'header_bg': 'gray',
            'font': 'black',
            'rowline': 'black',
            'colline': 'black',
            'red': 'red',
            'green': 'green',
        }
        _color.update(colors)
        _margin = self._position_tuple(*margin)

        table = table.copy()
        row_max_hei = [0] * len(table)
        col_max_wid = [0] * len(max(table, key=len))
        for i in range(len(table)):
            for j in range(len(table[i])):
                col_max_wid[j] = max(font.getsize(table[i][j])[0], col_max_wid[j])
                row_max_hei[i] = max(font.getsize(table[i][j])[1], row_max_hei[i])
        tab_width = sum(col_max_wid) + len(col_max_wid) * 2 * cell_pad[0]
        tab_heigh = sum(row_max_hei) + len(row_max_hei) * 2 * cell_pad[1]

        tab = Image.new('RGBA', (tab_width + _margin.left + _margin.right, tab_heigh + _margin.top + _margin.bottom),
                        _color['bg'])

        draw = ImageDraw.Draw(tab)

        draw.rectangle([(_margin.left, _margin.top), (_margin.left + tab_width, _margin.top + tab_heigh)],
                       fill=_color['cell_bg'], width=0)

        top = _margin.top
        for row_h in row_max_hei:
            draw.line([(_margin.left, top), (tab_width + _margin.left, top)], fill=_color['rowline'], width=width)
            top += row_h + cell_pad[1] * 2
        draw.line([(_margin.left, top), (tab_width + _margin.left, top)], fill=_color['rowline'], width=width)

        left = _margin.left
        for col_w in col_max_wid:
            draw.line([(left, _margin.top), (left, tab_heigh + _margin.top)], fill=_color['colline'], width=width)
            left += col_w + cell_pad[0] * 2
        draw.line([(left, _margin.top), (left, tab_heigh + _margin.top)], fill=_color['colline'], width=width)

        grid = np.zeros((len(self.text_generator.alphabet) + 1, tab_heigh + _margin.top, tab_width + _margin.left))
        grid[0] = grid[0] + 1

        top, left = _margin.top + cell_pad[1], 0
        for i in range(len(table)):
            left = _margin.left + cell_pad[0]
            for j in range(len(table[i])):
                _left = left
                if align and align[j] == 'c':
                    _left += (col_max_wid[j] - font.getsize(table[i][j])[0]) // 2
                elif align and align[j] == 'r':
                    _left += col_max_wid[j] - font.getsize(table[i][j])[0]
                crop, cm, text = self.text_generator.generate(text=table[i][j])
                tab.paste(crop, (_left, top), crop)
                grid = self._paste_grid_mask(cm, (_left, top), copy(grid))
                left += col_max_wid[j] + cell_pad[0] * 2
            top += row_max_hei[i] + cell_pad[1] * 2

        return tab, grid

    def _paste_grid_mask(self, grid_crop: np.ndarray,
                         left_top: Tuple[int, int], char_mask: np.ndarray) -> np.ndarray:
        for i in range(grid_crop.shape[0]):
            layer = Image.fromarray(grid_crop[i] * 255)
            layer = layer.convert('RGBA')

            if i == 0:
                back = Image.new('RGBA', layer.size, (255,) * 4)
            else:
                back = Image.new('RGBA', layer.size, (0,) * 4)

            layer = Image.composite(layer, back, layer)
            layer = layer.convert('1')
            char_mask[i, left_top[1]:left_top[1] + layer.size[1],
                      left_top[0]:left_top[0] + layer.size[0]] = np.array(layer)

        return char_mask
