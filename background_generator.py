import os
import random

from PIL import Image
from typing import List
from torch import autocast
from augmentators import *
from augmentators import BgAugmentBase
from diffusers import StableDiffusionPipeline


class BackgroundGenerator():
    def __init__(self, bg_size: List[int],
                 use_sd=False, bgs_path: str = None, augments: List[dict] = []) -> None:
        self.bg_size = bg_size
        self.use_sd = use_sd
        _bgs_formats = ('.png', 'jpg', 'jpeg', 'JPEG', 'jpeg', 'bmp')

        if use_sd:
            self._load_stable_diffusion()

        if bgs_path:
            if os.path.isdir(bgs_path):
                self.bgs_names = [os.path.join(bgs_path, b) for b in os.listdir(bgs_path) if b.endswith(_bgs_formats)]
            else:
                self.bgs_names = [bgs_path]
        else:
            self.bgs_names = []

        self.augments = augments

    def _load_stable_diffusion(self, model_path="CompVis/stable-diffusion-v1-4") -> None:
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(model_path,
                                                               use_auth_token=True,

                                                               ).to('cuda')
        self.sd_pipe.set_progress_bar_config(leave=False)
        self.prompt_chars = ' abcdefghijklmnopqrstuvwxyz'

    def generate(self, bg_path: str = None, augment: BgAugmentBase = None) -> Image.Image:
        if self.use_sd:
            sd_prompt = ''.join([self.prompt_chars[random.randint(0, len(self.prompt_chars) - 1)] for i in range(15)])

            with autocast('cuda'):
                bg = self.sd_pipe(sd_prompt, width=self.bg_size[0], height=self.bg_size[1])[0][0]

        elif bg_path:
            bg = Image.open(bg_path)
            bg = bg.resize(self.bg_size)

        elif self.bgs_names:
            bg = Image.open(random.choice(self.bgs_names))
            bg = bg.resize(self.bg_size)

        else:
            bg = Image.new(size=self.bg_size, mode='RGB', color=(255, 255, 255))

        if augment:
            bg = augment.augment(bg)

        elif self.augments:
            for aug in self.augments:
                aug = globals()[list(aug.keys())[0]](**list(aug.values())[0])
                bg = aug.augment(bg)

        bg = bg.convert('RGBA')
        return bg
