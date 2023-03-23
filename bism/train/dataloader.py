import torch
from torch import Tensor
import numpy as np
import glob
import os.path
import skimage.io as io
from typing import Dict, List, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple, Callable, List, Union, Optional

# basically just skoots

Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]


class dataset(Dataset):
    def __init__(self,
                 path: Union[List[str], str],
                 transforms: Optional[Transform] = lambda x: x,
                 pad_size: Optional[int] = 100,
                 device: Optional[str] = 'cpu',
                 sample_per_image: Optional[int] = 1):
        """

        Will output as device, but for all data to be stored on device, you must explicitly call self.to(device)

        :param path:
        :param transforms:
        :param pad_size:
        :param device:
        :param sample_per_image:
        """

        super(Dataset, self).__init__()

        # Reassigning variables
        self.files = []
        self.image = []
        self.centroids = []
        self.masks = []
        self.skeletons = []
        self.baked_skeleton = []
        self.transforms = transforms
        self.device = device
        self.pad_size: List[int] = [pad_size, pad_size]

        self._output_cache: List[Dict[str, Tensor]] = []

        self.sample_per_image = sample_per_image

        # Store all possible directories containing data in a list
        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f'{p}{os.sep}*.labels.tif'))

        for f in tqdm(self.files, desc='Loading Files: '):
            if os.path.exists(f[:-11:] + '.tif'):
                image_path = f[:-11:] + '.tif'
            else:
                raise FileNotFoundError(f'Could not find file: {image_path[:-4:]} with extensions .tif')

            skeleton = torch.load(f[:-11:] + '.skeletons.trch') if os.path.exists(
                f[:-11:] + '.skeletons.trch') else {-1: torch.tensor([])}

            image: np.array = io.imread(image_path)  # [Z, X, Y, C]
            masks: np.array = io.imread(f)  # [Z, X, Y]

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            masks: np.array = masks.transpose(1, 2, 0).astype(np.int32)

            scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
            scale = scale if image.max() > 1 else 1.

            # Convert to torch.tensor
            image: Tensor = torch.from_numpy(image / scale)  # .to(self.device)
            masks: Tensor = torch.from_numpy(masks).int().unsqueeze(0) # .to(self.device)

            # I need the images in a float, but use torch automated mixed precision so can store as half.
            # This may not be the same for you!
            self.image.append(image.half())
            self.masks.append(masks)


    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict = {'image': self.image[item],
                         'masks': self.masks[item]}

            # Transformation pipeline
            with torch.no_grad():
                # cache should always have something in it...
                data_dict = self.transforms(data_dict)  # Apply transforms and add to cache

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device, non_blocking=True)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {key: value.to(self.device, non_blocking=True) for (key, value) in data_dict[k].items()}

        return data_dict

    def to(self, device: str):
        """
        It is faster to do transforms on cuda, and if your GPU is big enough, everything can live there!!!
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]

        return self

    def cuda(self):
        self.to('cuda:0', non_blocking=True)
        return self

    def cpu(self):
        self.to('cpu')
        return self

    def pin_memory(self):
        """
        It is faster to do transforms on cuda, and if your GPU is big enough, everything can live there!!!
        """
        self.image = [x.pin_memory() for x in self.image]
        self.masks = [x.pin_memory() for x in self.masks]
        return self



class BackgroundDataset(Dataset):
    def __init__(self,
                 path: Union[List[str], str],
                 transforms: Optional[Transform] = lambda x: x,
                 pad_size: Optional[int] = 100,
                 device: Optional[str] = 'cpu',
                 sample_per_image: Optional[int] = 1):
        super(Dataset, self).__init__()
        """
        A dataset for images that contain nothing
        """

        # Reassigning variables
        self.files = []
        self.image = []
        self.transforms = transforms
        self.device = device

        path: List[str] = [path] if isinstance(path, str) else path

        for p in path:
            self.files.extend(glob.glob(f'{p}{os.sep}*.labels.tif'))

        for f in tqdm(self.files, desc='Loading Files: '):
            if os.path.exists(f[:-11:] + '.tif'):
                image_path = f[:-11:] + '.tif'
            else:
                raise FileNotFoundError(f'Could not find file: {image_path[:-4:]} with extensions .tif')

            skeleton = torch.load(f[:-11:] + '.skeletons.trch') if os.path.exists(
                f[:-11:] + '.skeletons.trch') else {-1: torch.tensor([])}

            image: np.array = io.imread(image_path)  # [Z, X, Y, C]
            masks: np.array = io.imread(f)  # [Z, X, Y]

            image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
            image: np.array = image.transpose(-1, 1, 2, 0)
            image: np.array = image[[2], ...] if image.shape[0] > 3 else image

            scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
            scale: int = scale if image.max() > 1 else 1

            image: Tensor = torch.from_numpy(image / scale) # .to(self.device)
            self.image.append(image.half())

    def __len__(self) -> int:
        return len(self.image) * self.sample_per_image

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        # We might artificially want to sample more times per image
        # Usefull when larging super large images with a lot of data.
        item = item // self.sample_per_image

        with torch.no_grad():
            data_dict = {'image': self.image[item],
                         'masks': torch.empty((1)),
                         'skeletons': {-1: torch.empty((1))},
                         'baked-skeleton': None}

            # Transformation pipeline
            with torch.no_grad():
                data_dict = self.transforms(data_dict)  # Apply transforms

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor):
                data_dict[k] = data_dict[k].to(self.device)
            elif isinstance(data_dict[k], dict):
                data_dict[k] = {key: value.to(self.device) for (key, value) in data_dict[k].items()}

        return data_dict

    def to(self, device: str):
        """
        It is faster to do transforms on cuda, and if your GPU is big enough, everything can live there!!!
        """
        self.image = [x.to(device) for x in self.image]
        self.masks = [x.to(device) for x in self.masks]
        self.skeletons = [{k: v.to(device) for (k, v) in x.items()} for x in self.skeletons]

        return self

    def cuda(self):
        self.to('cuda:0')
        return self

    def cpu(self):
        self.to('cpu')
        return self


class MultiDataset(Dataset):
    """
    Allows one to combine multiple datasets into one object. Itll behave the same way,
    but allows for very fine grain crontroll if your datasets come from many different places.

    for instance, if you have two large images and you want to sample them different amounts, you might do this:

    data0 = dataset(
                 path='/path/to/data0/,
                 transforms = lambda x: x,
                 device = 'cpu',
                 sample_per_image = 10)

    data1 = dataset(
                 path='/path/to/data1/,
                 transforms = lambda x: x,
                 device = 'cpu',
                 sample_per_image = 20) # <--- notice this here

    multidata = MultiDataset(data0, data1) # can now index this and access both data0 and data1

    """
    def __init__(self, *args):
        self.datasets: List[Dataset] = []
        for ds in args:
            if isinstance(ds, Dataset):
                self.datasets.append(ds)

        self._dataset_lengths = [len(ds) for ds in self.datasets]

        self.num_datasets = len(self.datasets)

        self._mapped_indicies = []
        for i, ds in enumerate(self.datasets):
            # range(len(ds)) necessary to not index whole dataset at start. SLOW!!!
            self._mapped_indicies.extend([i for _ in range(len(ds))])

    def __len__(self):
        return len(self._mapped_indicies)

    def __getitem__(self, item):
        i = self._mapped_indicies[item]  # Get the ind for the dataset
        _offset = sum(self._dataset_lengths[:i])  # Ind offset
        try:
            return self.datasets[i][item - _offset]
        except Exception:
            print(i, _offset, item - _offset, item, len(self.datasets[i]))
            raise RuntimeError

    def to(self, device: str):
        for i in range(self.num_datasets):
            self.datasets[i].to(device)
        return self

    def cuda(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cuda:0')
        return self

    def cpu(self):
        for i in range(self.num_datasets):
            self.datasets[i].to('cpu')
        return self


# Custom batching function!
def colate(data_dict: List[Dict[str, Tensor]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images = torch.stack([dd.pop('image') for dd in data_dict], dim=0)
    masks = torch.stack([dd.pop('masks') for dd in data_dict], dim=0)

    return images, masks

