import torch
import torchvision.transforms.functional as ttf
from torch import Tensor

from yacs.config import CfgNode
from typing import Dict, Optional


@torch.no_grad()
@torch.jit.ignore()
def transform_from_cfg(
    data_dict: Dict[str, Tensor], cfg: CfgNode, device: Optional[str] = None
) -> Dict[str, Tensor]:

    DEVICE: str = str(data_dict["image"].device) if device is None else device

    # Image should be in shape of [C, H, W, D]
    CROP_WIDTH = torch.tensor(cfg.AUGMENTATION.CROP_WIDTH, device=DEVICE)
    CROP_HEIGHT = torch.tensor(cfg.AUGMENTATION.CROP_HEIGHT, device=DEVICE)
    CROP_DEPTH = torch.tensor(cfg.AUGMENTATION.CROP_DEPTH, device=DEVICE)

    FLIP_RATE = torch.tensor(cfg.AUGMENTATION.FLIP_RATE, device=DEVICE)

    BRIGHTNESS_RATE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RATE, device=DEVICE)
    BRIGHTNESS_RANGE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RANGE, device=DEVICE)

    NOISE_GAMMA = torch.tensor(cfg.AUGMENTATION.NOISE_GAMMA, device=DEVICE)
    NOISE_RATE = torch.tensor(cfg.AUGMENTATION.NOISE_RATE, device=DEVICE)

    FILTER_RATE = torch.tensor(0.5, device=DEVICE)

    CONTRAST_RATE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RATE, device=DEVICE)
    CONTRAST_RANGE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RANGE, device=DEVICE)

    AFFINE_RATE = torch.tensor(cfg.AUGMENTATION.AFFINE_RATE, device=DEVICE)
    AFFINE_SCALE = torch.tensor(cfg.AUGMENTATION.AFFINE_SCALE, device=DEVICE)
    AFFINE_YAW = torch.tensor(cfg.AUGMENTATION.AFFINE_YAW, device=DEVICE)
    AFFINE_SHEAR = torch.tensor(cfg.AUGMENTATION.AFFINE_SHEAR, device=DEVICE)

    assert "masks" in data_dict, 'keyword "masks" not in data_dict'
    assert "image" in data_dict, 'keyword "image" not in data_dict'

    masks = data_dict["masks"]
    image = data_dict["image"]

    spatial_dims = masks.ndim - 1

    masks = masks.unsqueeze(-1) if masks.ndim == 3 else masks
    image = image.unsqueeze(-1) if image.ndim == 3 else image

    # ------------ Random Crop 1
    extra = 300
    w = (
        CROP_WIDTH + extra
        if CROP_WIDTH + extra <= image.shape[-3]
        else torch.tensor(image.shape[-3])
    )
    h = (
        CROP_HEIGHT + extra
        if CROP_HEIGHT + extra <= image.shape[-3]
        else torch.tensor(image.shape[-2])
    )
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[-1] else torch.tensor(image.shape[-1])

    # select a random point for croping
    x0 = torch.randint(0, image.shape[1] - w + 1, (1,), device=DEVICE)
    y0 = torch.randint(0, image.shape[2] - h + 1, (1,), device=DEVICE)
    z0 = torch.randint(0, image.shape[3] - d + 1, (1,), device=DEVICE)

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
    scale = scale if image.max() > 1 else 1.0

    image = image[:, x0:x1, y0:y1, z0:z1].to(DEVICE) / scale
    masks = masks[:, x0:x1, y0:y1, z0:z1].to(DEVICE)

    # -------------------affine (Cant use baked skeletons)
    if torch.rand(1, device=DEVICE) < AFFINE_RATE:
        angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
            1, device=DEVICE
        ) + AFFINE_YAW[0]
        shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
            1, device=DEVICE
        ) + AFFINE_SHEAR[0]
        scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
            1, device=DEVICE
        ) + AFFINE_SCALE[0]

        image = ttf.affine(
            image.permute(0, 3, 1, 2).float(),
            angle=angle.item(),
            shear=[float(shear)],
            scale=scale,
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        masks = ttf.affine(
            masks.permute(0, 3, 1, 2).float(),
            angle=angle.item(),
            shear=[float(shear)],
            scale=scale,
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

    # ------------ Center Crop 2
    w = CROP_WIDTH if CROP_WIDTH < image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT < image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH < image.shape[3] else torch.tensor(image.shape[3])

    # Center that instance
    x0 = torch.randint(0, image.shape[1] - w + 1, (1,), device=DEVICE)
    y0 = torch.randint(0, image.shape[2] - h + 1, (1,), device=DEVICE)
    z0 = torch.randint(0, image.shape[3] - d + 1, (1,), device=DEVICE)

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0:x1, y0:y1, z0:z1]
    masks = masks[:, x0:x1, y0:y1, z0:z1]

    # ------------------- x flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(1)
        masks = masks.flip(1)

    # ------------------- y flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(2)
        masks = masks.flip(2)

    # ------------------- z flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(3)
        masks = masks.flip(3)

    # # ------------------- Random Invert
    if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
        image = image.sub(1).mul(-1)

    # ------------------- Adjust Brightness
    if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
        # funky looking but FAST
        val = torch.empty(image.shape[0], device=DEVICE).uniform_(
            BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]
        )
        image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)

    # ------------------- Adjust Contrast
    if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
            (image.shape[0]), device=DEVICE
        ) + CONTRAST_RANGE[0]

        for z in range(image.shape[-1]):
            image[..., z] = ttf.adjust_contrast(image[..., z], contrast_val[0]).squeeze(
                0
            )

    # ------------------- Noise
    if torch.rand(1, device=DEVICE) < NOISE_RATE:
        noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
        image = image.add(noise).clamp(0, 1)

    if spatial_dims == 2:
        image = image[..., 0]
        masks = masks[..., 0]

    data_dict["image"] = image
    data_dict["masks"] = masks

    return data_dict
