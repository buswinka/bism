# bism - Biomedical Image Segmentation Models

BISM is a repository for training and evaluating biomedical instance segmentation models -- something akin to the `timm` package for 2D image tasks, but 3D instance segmentation. 
When at all possible, each model will offer a 2D or 3D implementation, however we will not provide pre-trained model files. 

No Documentation right now. In general, you launch a training run through a yaml configuration file. 
Check out `bism.train.__main__.py` as the starting point for training. `bism.config.config.py` for the default 
configuration for each approach. This should (hopefully) allow for repeatable training of 3D instance segmentation 
models of various types. 

To execute a training config, simply run `python bism/train --config_file "Path/To/Your/File.yaml"`.
To run a pretrained model, simply run `python bism/eval -m "path/to/model/file.trch" -i "path/to/image.tif"`
To launch the model inspector, run `python bism/gui`

This module is under active development so should not be used for anything but research purposes!

Current Models
---------------

| Model          | 2D  | 3D  | Scriptable |
|----------------|-----|-----|------------|
| UNet           | ✓   | ✓   | ✓          |
| UNeXT          | ✓   | ✓   | ✓          |
| Recurrent UNet | ✓   | ✓   | ✓          |
| Residual UNet  |     |     |            |
| Unet++         | ✓   | ✓   | ✓          |
| CellposeNet    | ✓   | ✓   | ✓          |


Current Generic Blocks
----------------------

| BLOCK NAME           | 2D   | 3D  |
|----------------------|------|-----|
| UNeXT Block          | ✓    | ✓   |
| ConcatConv           | ✓    | ✓   |
| Recurrent UNet BLock | ✓    | ✓   |
| Residual UNet BLock  | ✓    | ✓   |
| DropPath             | ✓    | ✓   |
| LayerNorm            | ✓    | ✓   |
| UpSample             | ✓    | ✓   |
| ViT Block            |      |     |

Segmentation Implementation
---------------------------

| APPROACH          | 2D | 3D |
|-------------------|----|----|
| Cellpose          |    |    |
| Affinities        |    | ✓  |
| Local Shape Desc. |    | ✓  |
| Omnipose          |    | ✓  |
| Auto Context LSD  |    | ✓  |
| Multitask LSDs    |    | ✓  |
| Semantic          | ✓  | ✓  |
| Mask RCNN         | ✓  |    |


Loss Functions
--------------
| Function         | Implemented |
|------------------|-------------|
| Dice             | ✓           |
| CL Dice          | ✓           |
| Tverksy          | ✓           |
| Jaccard          | ✓           |