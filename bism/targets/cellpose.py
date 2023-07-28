import torch
from torch import Tensor
import numpy as np
from bism.utils.morphology import binary_dilation
from tqdm import trange


def not_fucking_stuid_mask_to_flows(mask: Tensor, n_iter: int = 200) -> Tensor:
    device = mask.device
    dtype = mask.dtype

    unique = mask.unique()
    x, y, z = mask.shape
    flows = torch.zeros((x, y, z), dtype=torch.float32, device=device) #[x, y, z]

    for u in unique:
        if u == 0: continue
        _mask = mask == u

        center = _mask.nonzero().float().mean(0).round().long()
        assert mask[center[0], center[1], center[2]] == u, f'Center not in mask!'

        _last = torch.zeros((x, y, z), dtype=torch.float32, device=device)
        _last[center[0], center[1], center[2]] = 1.0

        for _ in range(n_iter):
            _out = binary_dilation(_last.unsqueeze(0).unsqueeze(0)).squeeze() * _mask

            if torch.all(_last == _out):
                break
            else:
                _last = _out

        flows.add_(_last)

    flows.div_(flows.max())

    return flows



#
#
# def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
#     """ runs diffusion on GPU to generate flows for training images or quality control
#
#     neighbors is 9 x pixels in masks,
#     centers are mask centers,
#     isneighbor is valid neighbor boolean 9 x pixels
#
#     """
#     if device is not None:
#         device = device
#     nimg = neighbors.shape[0] // 9
#     pt = torch.from_numpy(neighbors).to(device)
#
#     T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
#     meds = torch.from_numpy(centers.astype(int)).to(device).long()
#     isneigh = torch.from_numpy(isneighbor).to(device)
#     for i in range(n_iter):
#         T[:, meds[:, 0], meds[:, 1]] += 1
#         Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
#         Tneigh *= isneigh
#         T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)
#     del meds, isneigh, Tneigh
#     T = torch.log(1. + T)
#     # gradient positions
#     grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
#     del pt
#     dy = grads[:, 0] - grads[:, 1]
#     dx = grads[:, 2] - grads[:, 3]
#     del grads
#     mu_torch = np.stack((dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
#     return mu_torch
#
#
# def masks_to_flows_gpu(masks, device=None):
#     """ convert masks to flows using diffusion from center pixel
#     Center of masks where diffusion starts is defined using COM
#     Parameters
#     -------------
#     masks: int, 2D or 3D array
#         labelled masks 0=NO masks; 1,2,...=mask labels
#     Returns
#     -------------
#     mu: float, 3D or 4D array
#         flows in Y = mu[-2], flows in X = mu[-1].
#         if masks are 3D, flows in Z = mu[0].
#     mu_c: float, 2D or 3D array
#         for each pixel, the distance to the center of the mask
#         in which it resides
#     """
#     if device is None:
#         device = torch.device('cuda')
#
#     Ly0, Lx0 = masks.shape
#     Ly, Lx = Ly0 + 2, Lx0 + 2
#
#     masks_padded = np.zeros((Ly, Lx), np.int64)
#     masks_padded[1:-1, 1:-1] = masks
#
#     # get mask pixel neighbors
#     y, x = np.nonzero(masks_padded)
#     neighborsY = np.stack((y, y - 1, y + 1,
#                            y, y, y - 1,
#                            y - 1, y + 1, y + 1), axis=0)
#     neighborsX = np.stack((x, x, x,
#                            x - 1, x + 1, x - 1,
#                            x + 1, x - 1, x + 1), axis=0)
#     neighbors = np.stack((neighborsY, neighborsX), axis=-1)
#
#     # get mask centers
#     # slices = find_objects(masks)
#
#     centers = np.zeros((masks.max(), 2), 'int')
#     for i, si in enumerate(slices):
#         if si is not None:
#             sr, sc = si
#             ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
#             yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
#             yi = yi.astype(np.int32) + 1  # add padding
#             xi = xi.astype(np.int32) + 1  # add padding
#             ymed = np.median(yi)
#             xmed = np.median(xi)
#             imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
#             xmed = xi[imin]
#             ymed = yi[imin]
#             centers[i, 0] = ymed + sr.start
#             centers[i, 1] = xmed + sc.start
#
#     # get neighbor validator (not all neighbors are in same mask)
#     neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
#     isneighbor = neighbor_masks == neighbor_masks[0]
#     ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
#     n_iter = 2 * (ext.sum(axis=1)).max()
#     # run diffusion
#     mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx,
#                              n_iter=n_iter, device=device)
#
#     # normalize
#     mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)
#
#     # put into original image
#     mu0 = np.zeros((2, Ly0, Lx0))
#     mu0[:, y - 1, x - 1] = mu
#     mu_c = np.zeros_like(mu0)
#     return mu0, mu_c


if __name__ == '__main__':
    import skimage.io as io
    import matplotlib.pyplot as plt

    inst_mask = io.imread(
        '/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/hair-cell/chris/train/image-1.labels.tif')
    inst_mask = torch.from_numpy(inst_mask)[:, 200:300, 200:300]

    flows = not_fucking_stuid_mask_to_flows(inst_mask.cpu(), 100)

    print(flows.max(), flows.min())