import cv2
import random
import numpy as np
import math

from PIL import Image
from .bg_augment_base import BgAugmentBase
from barcode import EAN13
from barcode.writer import ImageWriter


class BarcodeAugment(BgAugmentBase):
    def __init__(self) -> None:
        self.barcode_squere = (0.01, 0.05)
        self.barcode_angle = (-2, 2)
        self.barcode_color = ((0, 250), (0, 250), (0, 250))

    def augment(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        code = ''.join([str(random.randint(0, 1)) for i in range(12)])
        barcode = EAN13(code, writer=ImageWriter()).render()
        _barcode_crop_points = (72, 7, 452, 193)
        barcode = barcode.crop(_barcode_crop_points)
        barcode = np.array(barcode)
        barcode = cv2.cvtColor(barcode, cv2.COLOR_RGB2BGRA)
        barcode[:, :, 3] = (barcode[:, :, 2] - 255) * -1
        barcode[:, :, 0] += random.randint(*self.barcode_color[0])
        barcode[:, :, 1] += random.randint(*self.barcode_color[1])
        barcode[:, :, 2] += random.randint(*self.barcode_color[2])
        barcode = Image.fromarray(barcode)
        barcode_squere = random.uniform(*self.barcode_squere) * image_size[0] * image_size[1]
        barcode_h = int(math.sqrt(barcode_squere * barcode.size[1] / barcode.size[0]))
        barcode_w = int(barcode_h * barcode.size[0] / barcode.size[1])
        barcode = barcode.resize((barcode_w, barcode_h))
        barcode = barcode.crop((0, 0, barcode.size[0], random.randint(int(barcode.size[1] * 0.3), barcode.size[1])))
        barcode = barcode.rotate(random.uniform(*self.barcode_angle), expand=True, resample=Image.BICUBIC)
        image.convert('RGBA')
        barcode_coord = (random.randint(0, image.size[0]), random.randint(0, image.size[1]))
        image.paste(barcode, barcode_coord, barcode)
        image = image.convert('RGB')
        return image


if __name__ == "__main__":
    bg = Image.open('JRrVZJYSkek.jpg')
    aug = BarcodeAugment()
    bg = aug.augment(bg)
    bg.show()
