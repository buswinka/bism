import torch
from typing import List
import skimage.io as io

from bism.utils.io import imread

def create_multi_crops(image_path: str,
                       relative_sizes: List[float] = [0.1, 0.2, 0.3, 0.4 , 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    creates crops of different size based on an original imgage for speed benchmarking...
    :param image_path:
    :return:
    """


    image: torch.Tensor = imread(image_path)
    assert image.shape[0] == 1

    image = image.squeeze(0)

    numel = image.numel()

    target_voxels = [torch.tensor(r * numel) for r in relative_sizes]
    print(numel, target_voxels)
    min_crop_size = (300, 300, 20) # 1800000 min

    x, y, z = image.shape
    shape_tensor = torch.tensor((x, y, z))

    assert (x * y * z) == numel
    """
    N = x * y * z  # can just do cubes until size gets too big...
    """
    for r, tv in zip(relative_sizes, target_voxels):
        _x, _y, _z = torch.tensor(0), torch.tensor(0), torch.tensor(0),

        assert tv < numel
        print(f'{tv=}, {image.shape}, {numel=}, {(_x*_y*_z)=}')
        i = 0
        while abs((_x * _y * _z) - tv) > 10000 and (_x * _y * _z) - tv < 0:
            # print(_x, _y, _z, _x*_y*_z, (_x * _y * _z) - numel)
            # fucking stupid...
            _x = min(x, _x + 10) - 1
            _y = min(y, _y + 10) - 1
            _z = min(z, _z + 10) - 1
            i+=1
            if i > 1000:
                raise ValueError(f'{_x=}, {_y=}, {_z=}, {image.shape}')

        crop = image[0:_x, 0:_y, 0:_z].to(torch.uint8)
        io.imsave(f'{image_path[:-4]}_{r}_crop.tif', crop.permute(2, 0, 1).numpy(), compression='zlib')


if __name__ == '__main__':
    path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Fig X - compare to affinity/data/benchmark/threeOHC_registered_8bit_cell2.tif'
    create_multi_crops(path)






