# bism - Biomedical Image Segmentation Models

This is a collection of generic PyTorch model constructors usefull for biomedical segmentation tasks. 
Something akin to the `timm` package for 2D image tasks. 
When at all possible, each model will offer a 2D or 3D implementation, however we will not provide pre-trained model files. 

Current Models
---------------

| Model          | 2D  | 3D  |
|----------------|-----|-----|
| UNet           | ✓   | ✓   |
| UNeXT          | ✓   | ✓   |
| Recurrent UNet | ✓   | ✓   |
| Residual UNet  |     |     |
| Unet++         | ✓   | ✓   |
| CPnet          | ✓   | ✓   |


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

