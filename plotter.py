import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from typing import Tuple


'''
Plotter(data_path)
plots and saves debug images with charmasks and words/rows coords

data_path folder should have 3 subfolders: "images", "row_coords", "char_masks"
also creates "plots" subfolder for save debug images
'''


class Plotter():
    def __init__(self, data_path: str) -> None:
        self.imgs_path = os.path.join(data_path, 'images')
        self.fb_path = os.path.join(data_path, 'row_coords')
        self.cm_path = os.path.join(data_path, 'char_masks')
        self.plot_path = os.path.join(data_path, 'plots')
        self.names = [name.replace('.png', '') for name in os.listdir(self.imgs_path) if name.endswith('png')]

        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)

    def int2rgb(self, RGBint: int) -> Tuple[int, int, int]:
        RGBint *= 123432
        blue = RGBint & 255
        green = (RGBint >> 8) & 255
        red = (RGBint >> 16) & 255
        return int(red), int(green), int(blue)

    def plot(self, filename: str = None,
             plot_rows: bool = True, plot_chars: bool = True) -> None:
        if filename:
            names = [filename]
        else:
            names = self.names

        for name in tqdm(names):
            img = cv2.imread(os.path.join(self.imgs_path, name + '.png'))

            if plot_chars:
                cm = np.load(os.path.join(self.cm_path, name + '.npy'))

                cm = np.round(cm)
                cm = np.argmax(cm, axis=0)
                cm2 = np.ones(img.shape)
                alpha = np.zeros(img.shape[:-1])

                for y in range(cm.shape[0]):
                    for x in range(cm.shape[1]):
                        r, g, b = self.int2rgb(cm[y][x])
                        cm2[y][x] = np.array([r, g, b])

                        if cm[y][x] != 0:
                            alpha[y][x] = 80

                cm2 = cm2.astype(int)
                cm2 = np.transpose(cm2, (2, 0, 1))
                cm2 = np.array([*cm2, alpha])
                cm2 = np.transpose(cm2, (1, 2, 0))

                y1, y2 = 0, 0 + cm2.shape[0]
                x1, x2 = 0, 0 + cm2.shape[1]

                alpha_s = cm2[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (alpha_s * cm2[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

            if plot_rows:
                with open(os.path.join(self.fb_path, name + '.json'), 'r', encoding='utf-8') as jf:
                    fb = json.load(jf)

                for item in fb['rows']:
                    right_bot = (item['left_top'][0] + item['size'][1],
                                 item['left_top'][1] + item['size'][0])
                    cv2.rectangle(img, item['left_top'], right_bot, color=(0, 255, 0), thickness=1)

                tright_bot = (fb['table']['left_top'][0] + fb['table']['size'][1],
                              fb['table']['left_top'][1] + fb['table']['size'][0])
                cv2.rectangle(img, fb['table']['left_top'], tright_bot, color=(0, 0, 255), thickness=2)

            cv2.imwrite(os.path.join(self.plot_path, name + '.png'), img)


if __name__ == "__main__":
    plotter = Plotter('gen_data')
    plotter.plot(plot_rows=True, plot_chars=True)
