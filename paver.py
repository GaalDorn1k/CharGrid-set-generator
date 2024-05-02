import random
import numpy as np

from copy import copy
from typing import Tuple, Union


'''
Paver(H, W)
solves the problem of non-optimal tiling of a H x W field with random rectangles along random coordinates

Paver.get_random_coords(H, W)
returns the coords of the H x W rect placement on the field if it possible
fills the field with input rectangles
if there is no space left on the field returns None
'''


class Paver():
    def __init__(self, h: int, w: int, i: int) -> None:
        self.i = i
        self.field = np.zeros((h, w))
        self.coords = np.where(self.field == 0)
        self.coords2 = []
        self.color = 1
        self.delta = 10

    def get_random_coords(self, w: int, h: int) -> Union[Tuple[int, int], None]:
        poly = np.zeros((h, w))
        poly = poly + self.color
        field2 = copy(self.field)
        field2[-h:] = 1
        field2[:, -w:] = 1

        for c in self.coords2:
            y = c[0] - h if c[0] - h > 0 else 0
            x = c[1] - w if c[1] - w > 0 else 0
            field2[y:c[2], x:c[3]] = 2

        self.coords = np.where(field2 == 0)

        if len(self.coords[0]) == 0:
            return None

        index = random.randint(0, len(self.coords[0]) - 1)
        y = self.coords[0][index]
        x = self.coords[1][index]
        self.field[y:y + h, x:x + w] += poly
        self.coords2.append([y - self.delta,
                             x - self.delta,
                             y + h + self.delta,
                             x + w + self.delta])
        self.color += 1
        return x, y
