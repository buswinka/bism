# bism - Biomedical Image Segmentation Models

This is a collection of generic PyTorch model constructors useful for biomedical segmentation tasks. 
Something akin to the `timm` package for 2D image tasks, but 3D instance segmentation. 
When at all possible, each model will offer a 2D or 3D implementation, however we will not provide pre-trained model files. 

No Documentation right now. In general, you launch a training run through a yaml configuration file. 
Check out bism.train.__main__.py as the starting point for training. bism.config.config.py for the defualt 
configuration for each approach. This should (hopefully) allow for repeatable training of 3D instance segmentation 
models of various types.

Current Models
---------------

| Model          | 2D  | 3D  | Scriptable |
|----------------|-----|-----|------------|
| UNet           | ✓   | ✓   | ✓          |
| UNeXT          | ✓   | ✓   | ✓          |
| Recurrent UNet | ✓   | ✓   | ✓          |
| Residual UNet  |     |     |            |
| Unet++         | ✓   | ✓   | ✓          |
| CPnet          | ✓   | ✓   | ✓          |


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

| APPROACH          | 2D | 3D  |
|-------------------|----|-----|
| Cellpose          |    |     |
| Affinities        |    | ✓   |
| Local Shape Desc. |    | ✓   |
| Omnipose          |    | ✓   |
| Auto Context LSD  |    | ✓   |
| Multitask LSDs    |    | ✓   |


Loss Functions
--------------
| Function         | Implemented |
|------------------|-------------|
| Dice             | ✓           |
| CL Dice          | ✓           |
| Tverksy          | ✓           |
| Jaccard          | ✓           |