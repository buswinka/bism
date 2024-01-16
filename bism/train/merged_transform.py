import random
from typing import Dict, Callable

import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
from torch import Tensor
from yacs.config import CfgNode


def transform_from_cfg(*args, **kwargs):
    raise RuntimeError('Depreciated function. Please refactor to use bism.train.merged_transform.TransformFromCfg()')

class TransformFromCfg(nn.Module):
    def __init__(self, cfg: CfgNode, device: torch.device, scale: float = 255.0):
        super(TransformFromCfg, self).__init__()
        """
        Why? Apparently a huge amount of overhead is just initializing this from cfg
        If we preinitalize, then we can save on overhead, to do this, we need a class...
        Probably a reasonalbe functional way to do this. Ill think on it later
        
        """

        self.prefix_function = self._identity
        self.posfix_function = self._identity

        self.dataset_mean = None
        self.dataset_std = None

        self.cfg = cfg

        self.DEVICE = device
        self.SCALE = scale

        self.CROP_WIDTH = cfg.AUGMENTATION.CROP_WIDTH
        self.CROP_HEIGHT = cfg.AUGMENTATION.CROP_HEIGHT

        self.CROP_DEPTH = cfg.AUGMENTATION.CROP_DEPTH

        self.FLIP_RATE = cfg.AUGMENTATION.FLIP_RATE

        self.BRIGHTNESS_RATE = cfg.AUGMENTATION.BRIGHTNESS_RATE
        self.BRIGHTNESS_RANGE = cfg.AUGMENTATION.BRIGHTNESS_RANGE
        self.NOISE_GAMMA = cfg.AUGMENTATION.NOISE_GAMMA
        self.NOISE_RATE = cfg.AUGMENTATION.NOISE_RATE

        self.FILTER_RATE = 0.5

        self.CONTRAST_RATE = cfg.AUGMENTATION.CONTRAST_RATE
        self.CONTRAST_RANGE = cfg.AUGMENTATION.CONTRAST_RANGE

        self.AFFINE_RATE = cfg.AUGMENTATION.AFFINE_RATE
        self.AFFINE_SCALE = cfg.AUGMENTATION.AFFINE_SCALE
        self.AFFINE_SHEAR = cfg.AUGMENTATION.AFFINE_SHEAR
        self.AFFINE_YAW = cfg.AUGMENTATION.AFFINE_YAW

    def _identity(self, *args):
        return args if len(args) > 1 else args[0]

    def _crop1(self, image, masks):
        masks = masks.unsqueeze(-1) if masks.ndim == 3 else masks
        image = image.unsqueeze(-1) if image.ndim == 3 else image

        C, X, Y, Z = image.shape
        # ------------ Random Crop 1
        extra = 300
        w = self.CROP_WIDTH + extra if self.CROP_WIDTH + extra <= X else X
        h = self.CROP_HEIGHT + extra if self.CROP_HEIGHT + extra <= Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH <= Z else Z

        # select a random point for croping
        x0 = random.randint(0, X - w)
        y0 = random.randint(0, Y - h)
        z0 = random.randint(0, Z - d)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        image = image[:, x0:x1, y0:y1, z0:z1]
        masks = masks[:, x0:x1, y0:y1, z0:z1]

        if image.device != self.DEVICE:
            image = image.to(self.DEVICE)

        if masks.device != self.DEVICE:
            masks = masks.to(self.DEVICE)

        return image, masks

    def _affine(self, image, masks):
        angle = random.uniform(*self.AFFINE_YAW)
        shear = random.uniform(*self.AFFINE_YAW)
        scale = random.uniform(*self.AFFINE_SCALE)

        image = ttf.affine(
            image.permute(0, 3, 1, 2).float(),
            angle=angle,
            shear=[float(shear)],
            scale=scale,
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        masks = ttf.affine(
            masks.permute(0, 3, 1, 2).float(),
            angle=angle,
            shear=[float(shear)],
            scale=scale,
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        return image, masks

    def _crop2(self, image, masks):
        C, X, Y, Z = image.shape
        w = self.CROP_WIDTH if self.CROP_WIDTH < X else X
        h = self.CROP_HEIGHT if self.CROP_HEIGHT < Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH < Z else Z

        # Center that instance
        x0 = random.randint(0, X - w)
        y0 = random.randint(0, Y - h)
        z0 = random.randint(0, Z - d)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        image = image[:, x0:x1, y0:y1, z0:z1]
        masks = masks[:, x0:x1, y0:y1, z0:z1]

        return image, masks

    def _flipX(self, image, masks):
        image = image.flip(1)
        masks = masks.flip(1)
        return image, masks

    def _flipY(self, image, masks):
        image = image.flip(2)
        masks = masks.flip(2)
        return image, masks

    def _flipZ(self, image, masks):
        image = image.flip(3)
        masks = masks.flip(3)
        return image, masks

    def _invert(self, image, masks):
        image.sub_(1).mul_(-1)
        return image, masks

    def _brightness(self, image, masks):
        val = random.uniform(*self.BRIGHTNESS_RANGE)
        # in place ok because flip always returns a copy
        image = image.add(val)
        return image, masks

    def _contrast(self, image, masks):
        contrast_val = random.uniform(*self.CONTRAST_RANGE)
        # [ C, X, Y, Z ] -> [Z, C, X, Y]
        image = ttf.adjust_contrast(image.permute(3, 0, 1, 2), contrast_val).permute(1,2,3,0)

        return image, masks

    def _noise(self, image, masks):
        noise = torch.rand(image.shape, device=self.DEVICE) * self.NOISE_GAMMA
        image = image.add(noise)
        return image, masks

    def _normalize(self, image, masks):
        # mean = image.float().mean()
        # std = image.float().std()
        mean = image.float().mean() if not self.dataset_mean else self.dataset_mean
        std = image.float().std() if not self.dataset_std else self.dataset_std

        image = image.float().sub(mean).div(std)
        return image, masks

    def set_dataset_mean(self, mean):
        self.dataset_mean = mean
        return self

    def set_dataset_std(self, std):
        self.dataset_std = std
        return self

    @torch.no_grad()
    def forward(self, data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:

        assert "masks" in data_dict, 'keyword "masks" not in data_dict'
        assert "image" in data_dict, 'keyword "image" not in data_dict'

        data_dict = self.prefix_function(data_dict)

        masks = data_dict["masks"]
        image = data_dict["image"]

        spatial_dims = masks.ndim - 1

        image, masks = self._crop1(image, masks)

        # data_dict['masks'] = masks
        # data_dict['image'] = image

        # data_dict = self.postcrop_function(data_dict)
        #
        # masks = data_dict["masks"]
        # image = data_dict["image"]
        #

        # scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
        # scale = scale if image.max() > 1 else 1.0

        image, masks = self._normalize(image, masks)

        # ------------ Center Crop 2
        image, masks = self._crop2(image, masks)

        # ------------------- x flip
        if random.random() < self.FLIP_RATE:
            image, masks = self._flipX(image, masks)

        # ------------------- y flip
        if random.random() < self.FLIP_RATE:
            image, masks = self._flipY(image, masks)

        # ------------------- z flip
        if random.random() < self.FLIP_RATE:
            image, masks = self._flipZ(image, masks)

        # # ------------------- Random Invert
        if random.random() < self.BRIGHTNESS_RATE:
            image, masks = self._invert(image, masks)

        # ------------------- Adjust Brightness
        if random.random() < self.BRIGHTNESS_RATE:
            image, masks = self._brightness(image, masks)

        # ------------------- Adjust Contrast
        if random.random() < self.CONTRAST_RATE:
            image, masks = self._contrast(image, masks)

        # ------------------- Noise
        if random.random() < self.NOISE_RATE:
            image, masks = self._noise(image, masks)


        if spatial_dims == 2:
            image = image[..., 0]
            masks = masks[..., 0]

        data_dict["image"] = image
        data_dict["masks"] = masks

        # assert image.ndim==4
        # assert masks.ndim==4

        data_dict = self.posfix_function(data_dict)

        return data_dict

    def pre_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.prefix_function = fn
        return self

    def post_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.posfix_function = fn
        return self

    def post_crop_fn(self, fn):
        self.postcrop_function = fn
        return self

    def __repr__(self):
        return f"TransformFromCfg[Device:{self.DEVICE}]\ncfg.AUGMENTATION:\n=================\n{self.cfg.AUGMENTATION}]"
