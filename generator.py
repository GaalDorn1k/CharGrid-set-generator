import os
import json
import random
import numpy as np

from copy import copy
from PIL import Image
from tqdm import tqdm
from paver import Paver
from typing import Tuple
from row_generator import RowGenerator
from table_generator import TableGenerator
from background_generator import BackgroundGenerator


class Generator():
    def __init__(self, config: dict) -> None:
        self.num_imgs = config['Generator']['num_imgs']
        save_path = config['Generator']['save_path']
        self.alphabet = config['Generator']['alphabet']
        self.row_angle = config['Generator']['row_angle']
        self.words_in_page = config['Generator']['words_in_page']
        self.bg_generator = BackgroundGenerator(**config['BackgroundGenerator'])
        self.crop_generator = RowGenerator(**config['RowGenerator'], alphabet=self.alphabet)
        self.table_generator = TableGenerator(**config['TableGenerator'],
                                              text_generator=self.crop_generator,
                                              size_range=config['RowGenerator']['font_size'])
        self.save_subfolders = ['images', 'field_masks', 'char_masks', 'row_coords']

        for i in range(len(self.save_subfolders)):
            subfolder = os.path.join(save_path, self.save_subfolders[i])
            self.save_subfolders[i] = subfolder
            if not os.path.exists(subfolder):
                os.mkdir(subfolder)

    def _paste_grid_mask(self, grid_crop: np.ndarray,
                         angle: float, left_top: Tuple[int, int], char_mask: np.ndarray) -> np.ndarray:
        for i in range(grid_crop.shape[0]):
            layer = Image.fromarray(grid_crop[i] * 255)
            layer = layer.convert('RGBA')
            layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)

            if i == 0:
                back = Image.new('RGBA', layer.size, (255,) * 4)
            else:
                back = Image.new('RGBA', layer.size, (0,) * 4)

            layer = Image.composite(layer, back, layer)
            layer = layer.convert('1')
            char_mask[i, left_top[1]:left_top[1] + layer.size[1],
                      left_top[0]:left_top[0] + layer.size[0]] = np.array(layer)

        return char_mask

    def _paste_field_mask(self, field_mask: np.ndarray, left_top: Tuple[int, int],
                          field_size: Tuple[int, int], angle: float, field_code: int) -> Tuple[np.ndarray, tuple]:
        field = Image.new('RGBA', (field_size[1], field_size[0]), 1)
        layer = Image.new('1', (field_mask.shape[2], field_mask.shape[1]), 0)
        field = field.rotate(angle, expand=True, resample=Image.BICUBIC)
        back = Image.new('RGBA', field.size, (255,) * 4)
        field = Image.composite(field, back, field)
        field = field.convert('1')
        layer.paste(field, left_top)
        layer = np.array(layer)
        field_mask[0] -= layer
        field_mask[field_code + 1] += layer
        return field_mask, field.size

    def generate(self) -> None:
        imgs_path = self.save_subfolders[0]
        exists_imgs = [f.replace('.png', '') for f in os.listdir(imgs_path) if f.endswith('png')]
        exists_imgs = [int(n) for n in exists_imgs if n.isdigit()]
        first_name = max(exists_imgs) + 1 if exists_imgs else 0

        for i in tqdm(range(0, self.num_imgs)):
            bg = self.bg_generator.generate()
            cm = np.zeros((len(self.alphabet) + 1, bg.size[1], bg.size[0]))
            cm[0] = cm[0] + 1
            fm = np.zeros((2, bg.size[1], bg.size[0]))
            fm[0] = fm[0] + 1
            size = list(bg.size)
            size.reverse()
            paver = Paver(*size, i)
            row_coords = []
            max_words = random.randint(*self.words_in_page)
            words_count = -1

            table, tgrid, tdata = self.table_generator.generate()
            tcoords = paver.get_random_coords(*table.size)
            cm = self._paste_grid_mask(tgrid, 0, tcoords, copy(cm))

            if tcoords:
                bg.paste(table, tcoords, table)

            while True:
                words_count += 1
                text_crop, grid_crop, text = self.crop_generator.generate()
                fmm = copy(fm)
                cmm = copy(cm)
                angle = random.uniform(*self.row_angle)
                text_crop = text_crop.rotate(angle, expand=True, resample=Image.BICUBIC)
                coords = paver.get_random_coords(*text_crop.size)

                if coords is None or words_count == max_words:
                    break

                bg.paste(text_crop, coords, text_crop)
                cm = self._paste_grid_mask(np.array(grid_crop), angle, coords, cmm)
                fm, rotate_field_size = self._paste_field_mask(fmm, coords, grid_crop.shape[1:], angle, 0)
                row_coords.append({'left_top': [int(c) for c in coords],
                                   'size': [rotate_field_size[1], rotate_field_size[0]],
                                   'text': text})

            bg = bg.convert('RGB')
            name = first_name + i
            bg.save(os.path.join(self.save_subfolders[0], f'{name}.png'))
            np.save(os.path.join(self.save_subfolders[1], f'{name}.npy'), fm)
            np.save(os.path.join(self.save_subfolders[2], f'{name}.npy'), cm)

            tcoords = {'left_top': [int(c) for c in tcoords],
                       'size': [table.size[1], table.size[0]],
                       'text': tdata}

            with open(os.path.join(self.save_subfolders[3], f'{name}.json'), 'w', encoding='utf-8') as jf:
                json.dump({'rows': row_coords, 'table': tcoords}, jf, indent=4, ensure_ascii=False)
