from typing import Tuple, Optional
import random
import torch
from torch.nn import functional as F
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors  # for_ custom colormap
from scipy import ndimage  # for: getting boundaries of image


def normalize(tensor):
    return (tensor * 2) - 1  # map [0,1]->[-1,1]


def denormalize(tensor):
    return (tensor + 1) / 2  # map [-1,1]->[0,1]


def pad3d(size, image, max_bbox=None) -> Tuple[torch.Tensor, Optional[Tuple[slice, slice, slice]]]:
    image = torch.Tensor(image)
    d, w, h = image.shape[-3], image.shape[-2], image.shape[-1]
    d_max, w_max, h_max = size
    d_pad = max((d_max - d) / 2, 0)
    w_pad = max((w_max - w) / 2, 0)
    h_pad = max((h_max - h) / 2, 0)
    padding = (
        int(floor(h_pad)),
        int(ceil(h_pad)),
        int(floor(w_pad)),
        int(ceil(w_pad)),
        int(floor(d_pad)),
        int(ceil(d_pad)),
    )
    x = F.pad(image, padding, value=0, mode="constant")

    if max_bbox is not None:
        max_bbox = list(max_bbox)
        for i, s in enumerate(max_bbox, 1):
            s: slice
            pad_e = padding[2 * -i + 1]
            pad_s = padding[2 * -i]
            max_bbox[i - 1] = slice(s.start - pad_s, s.stop + pad_e)
        max_bbox = tuple(max_bbox)
    return x, max_bbox


def compute_bbox(array, minimum=0, dist=0, zooms=(1, 1, 1)):
    """
    Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

    Args:
        minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
        dist (int): The amount of padding to be added to the cropped image. Default value is 0.
    Returns:
        ex_slice: A tuple of slice objects that need to be applied to crop the image.
        origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

    Note:
        - The computed slice removes the unused space from the image based on the minimum value.
        - The padding is added to the computed slice.
        - If the computed slice reduces the array size to zero, a ValueError is raised.
    """
    shp = array.shape
    d = np.around(dist / np.asarray(zooms)).astype(int)
    msk_bin = np.zeros(array.shape, dtype=bool)
    msk_bin[array > minimum] = 1
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)

    if cor_msk[0].shape[0] == 0:
        raise ValueError("Array would be reduced to zero size")

    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
    y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
    z0 = c_min[2] - d[2] if (c_min[2] - d[2]) > 0 else 0
    x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[0] else shp[0]
    y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[1] else shp[1]
    z1 = c_max[2] + d[2] if (c_max[2] + d[2]) < shp[2] else shp[2]

    bbox = tuple([slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1)])
    # bbox = ((x0, x1 + 1), (y0, y1 + 1), (z0, z1 + 1))

    return bbox


def random_crop(target_shape: Tuple[int, int, int], *arrs: torch.Tensor):
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)


def get_latest_Checkpoint(opt, version="*", log_dir_name="lightning_logs", best=False):
    import glob
    import os

    ckpt = "*"
    if best:
        ckpt = "*best*"
    checkpoints = None
    if isinstance(opt, str) or not opt.new:
        if isinstance(opt, str):
            checkpoints = sorted(glob.glob(f"{log_dir_name}/{opt}/version_{version}/checkpoints/{ckpt}.ckpt"), key=os.path.getmtime)

        else:
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt.experiment_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )

        if len(checkpoints) == 0:
            checkpoints = None
        else:
            checkpoints = checkpoints[-1]
    else:
        return None

    return checkpoints


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


#### Plotting ####


# helper function to plot transparent mask with boundary on given axis
def _plot_mask(axis, mask, color=(0.90, 0.60, 0.0, 1.0), alpha_mask=0.5, alpha_boundary=1.0):
    cmap = colors.ListedColormap([(0, 0, 0, 0), color])  # zeros: transparent  # ones: provided_color
    axis.imshow(mask, cmap=cmap, alpha=alpha_mask, interpolation="nearest")
    boundary = ndimage.morphology.binary_dilation(mask, np.ones((3, 3), dtype=bool)) != mask  # get boundaries
    axis.imshow(boundary, cmap=cmap, alpha=alpha_boundary)


# plot single image, 5 slices
def plot_3D(image, healthyMask=None, unhealthyMask=None, generalMask=None, steps_size=15, cmap_image="gray", dpi=100):
    """Input might be CDWH or DWH.

    COmputes center slice

    Args: [TODO]
        mask (_type_): _description_
        steps_size (int, optional): _description_. Defaults to 15.
        cmap_image (str, optional): _description_. Defaults to "gray".
    """
    shape = image.shape

    # sliceIDs = list(range(0, shape[-1], steps_size)) #all slices with distance steps_size
    middle = int(shape[-1] / 2)  # middle slice index
    sliceIDs = [middle - 2 * steps_size, middle - steps_size, middle, middle + steps_size, middle + 2 * steps_size]
    size = (shape[-3] / 60.0, shape[-2] / 60.0)  # D/120, W/120

    # Reduce dimensions CWDH -> WDH if necessary

    fig, ax = plt.subplots(figsize=(size[0] * len(sliceIDs), size[1]), nrows=1, ncols=len(sliceIDs), squeeze=True, sharey=True, dpi=dpi)

    for i, sliceID in enumerate(sliceIDs):
        # plot image
        # ax[i].set_title(sliceID)
        img = image[0, :, :, sliceID].T if len(shape) == 4 else image[:, :, sliceID].T
        ax[i].imshow(img, cmap=cmap_image, interpolation="nearest")

        # plot masks if provided
        if not healthyMask is None:
            img = healthyMask[0, :, :, sliceID].T if len(healthyMask.shape) == 4 else healthyMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.0, 0.80, 0.20, 1.0))  # green

        if not unhealthyMask is None:
            img = unhealthyMask[0, :, :, sliceID].T if len(unhealthyMask.shape) == 4 else unhealthyMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.90, 0.10, 0.10, 1.0))  # red

        if not generalMask is None:
            img = generalMask[0, :, :, sliceID].T if len(generalMask.shape) == 4 else generalMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.90, 0.60, 0.0, 1.0))  # orange

    plt.show()
