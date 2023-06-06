# @author Robert Graf; deep-spine.de 2023
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from baseline_utils import compute_bbox, normalize, pad3d, random_crop
from math import ceil


class Dataset_Inference(Dataset):
    """Dataset for Inference purposes.

    Contains no ground truth (t1n), just its voided version (t1n-voided) as well as the mask used for voiding (mask).
    This mask cuts away the tumor tissue as well as some healthy tissue. Which cutout is from which tissue type is not
    reversible.

    Args:
        root_dir: Path to dataset files.
        crop_shape: Cuboid shape the images will be reduced to. Set to (240,240,155) to lose no data.
        center_on_mask: Center cropped cuboid at the mask.
    Raises:
        UserWarning: If your dataset does not contain files having the proper shape (240,240,155)

    Returns:
        __getitem__: Returns a dictionary containing:
            "voided_image": The cropped version of t1n-voided.
            "t1n_voided_path": Path to t1n-voided (relevant for reversing the sample)
            "mask": Cropped version of mask.
            "cropped_bbox": Bounding box that was used for cropping
            "max_v": Maximal value of t1 image (used for normalization)
            "name": Sample ID. e.g. "BraTS-GLI-00003-000"
    """

    def __init__(self, root_dir: Path, crop_shape=(128, 128, 96), center_on_mask=False):
        # Initialize variables
        self.root_dir = root_dir
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask

        # Get dataset paths
        # multiline:           list(root_dir.rglob("**/BraTS-GLI-*-*-t1n-????.nii.gz"))
        self.list_paths_t1n_voided = list(root_dir.rglob("**/BraTS-GLI-*-*-t1n-voided.nii.gz"))
        self.list_paths_mask = list(root_dir.rglob("**/BraTS-GLI-*-*-mask.nii.gz"))
        # self.list_paths_mask_unhealthy = list(root_dir.rglob("**/BraTS-GLI-*-*-mask-unhealthy.nii.gz"))

    def __len__(self):
        return len(self.list_paths_mask)

    def preprocess(self, t1n_voided: np.ndarray, mask: np.ndarray):
        """Transforms the images to a more unified format.

        Normalizes to -1,1. Crops to bounding box. Adds additional dimension in front for "channel".

        Args:
            t1n_voided (np.ndarray): t1n-voided from t1n-voided file.
            mask (np.ndarray): mask from mask file.

        Raises:
            UserWarning: When your input images are not (240, 240, 155)

        Returns:
            t1n_voided_max_v: Maximal value of t1 image (used for normalization)
            crop_box: Bounding box that was used for cropping
            t1n_voided_crop: The cropped version of t1n-voided
            mask_crop: Cropped version of mask.
        """

        # Size Assertions
        referenceShape = (240, 240, 155)
        if t1n_voided.shape != referenceShape or mask.shape != referenceShape:
            raise UserWarning(f"Your t1n-voided or mask shape is not {referenceShape}, they are: {t1n_voided.shape} and {mask.shape}")

        # Normalize the image to [0,1]
        t1n_voided[t1n_voided < 0] = 0  # Values below 0 are considered to be noise.
        # Note that only 4 samples fulfill min(t1)!=0 : GLI-01332-000, GLI-00048-001, GLI-00446-000 and BraTS2023_01655
        t1n_voided_max_v = np.max(t1n_voided)
        t1n_voided /= t1n_voided_max_v

        # Crop to smaller size
        if self.center_on_mask:  # crop to region aground target
            shape = mask.shape[-3:]
            min_bbox = compute_bbox(mask)
            max_bbox = []
            for i, s in enumerate(min_bbox):
                s: slice
                d = self.crop_shape[i] - (s.stop - s.start)
                s_n = slice(s.start - d // 2, s.stop + ceil(d / 2))
                if s_n.start < 0:
                    s_n = slice(0, self.crop_shape[i])
                if s_n.stop > shape[i]:
                    s_n = slice(shape[i] - self.crop_shape[i], shape[i])

                max_bbox.append(s_n)
            max_bbox = tuple(max_bbox)
        else:  # crop to whole brain (removes everything empty space)
            max_bbox = compute_bbox(t1n_voided)

        # apply crop
        t1n_voided_crop = t1n_voided[max_bbox]
        mask_crop = mask[max_bbox]

        # pad if too small
        t1n_voided_crop, crop_box = pad3d(self.crop_shape, t1n_voided_crop, max_bbox)
        mask_crop, _ = pad3d(self.crop_shape, mask_crop)

        # crop if bigger
        t1n_voided_crop, mask_crop = random_crop(self.crop_shape, t1n_voided_crop, mask_crop)

        # map t1n images to -1,1
        t1n_voided_crop = normalize(t1n_voided_crop)

        # Note: you might want do some data augmentation here (possible TODO)

        # add dimension for channel (we only have one though)
        t1n_voided_crop = t1n_voided_crop.unsqueeze(0)
        mask_crop = mask_crop.unsqueeze(0)

        # set fitting datatype
        mask_crop = mask_crop.bool()

        return t1n_voided_max_v, crop_box, t1n_voided_crop, mask_crop

    def __getitem__(self, idx):
        # Inference - target: mask
        mask_path = self.list_paths_mask[idx]
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata()

        # Inference - context/input image: t1n_voided
        t1n_voided_path = self.list_paths_t1n_voided[idx]
        t1n_voided_img = nib.load(t1n_voided_path)
        t1n_voided = t1n_voided_img.get_fdata()

        # Crop and normalize images
        t1n_voided_max_v, crop_box, t1n_voided_crop, mask_crop = self.preprocess(t1n_voided, mask)

        # Output/Sample data
        sample_dict = {
            "voided_image": t1n_voided_crop,
            "t1n_voided_path": str(t1n_voided_path),  # path to the t1n-voided file for this sample.
            "mask": mask_crop,
            "cropped_bbox": str(crop_box),  # bounding be used to crop
            # TODO: cropped_bbox CANNOT be given to dataloader/network directly (collate_fn -> type error)
            # currently we do str <-> eval conversion which is ugly.
            "max_v": t1n_voided_max_v,  # maximal t1n_voided value used for normalization
            "name": mask_path.name[:19],
        }

        return sample_dict

    @classmethod
    def get_result_image(cls, prediction, sample):
        """Reverse the transformations that were necessary for the network.

        Args:
            prediction : numpy.ndarray
                The network output ( from model.forward() ).
            sample : dict
                The sample used as input for the network output.

        Returns:
            result : numpy.ndarray
                The resulting full scale inferred image
            img : nib.nifti1.Nifti1Image
                The same as "result" but as proper nifti file that can be saved to the file system
        """

        # Get target image
        t1n_voided_img = nib.load(sample["t1n_voided_path"])  # get target
        affine = t1n_voided_img.affine
        t1n_voided = t1n_voided_img.get_fdata()

        # Remove channels
        prediction = prediction[0]

        # Reverse sample normalize
        prediction = (prediction + 1) / 2

        # reverse scaling
        prediction = prediction * sample["max_v"]

        # limit output to the inpainting region
        mask = sample["mask"][0]  # cropped mask, no channel
        prediction_minimal = np.zeros_like(prediction)
        prediction_minimal[mask] = prediction[mask]

        # reverse cropping
        bb = eval(sample["cropped_bbox"])
        prediction_full = np.zeros_like(t1n_voided)
        prediction_full[bb] = prediction_minimal

        # Place output in the inpainting regions
        result = t1n_voided + prediction_full

        # create output nifti image
        img = nib.nifti1.Nifti1Image(result, affine)

        return result, img


class Dataset_Training(Dataset):
    """Dataset for Training purposes.

    Contains ground truth (t1n). Also contains a voided image where only the healthy tissue is voided (t1n +
    mask-healthy). The masked used for voiding (mask-healthy) is also present.

    Args:
        root_dir: Path to dataset files.
        crop_shape: Cuboid shape the images will be reduced to. Set to (240,240,155) to lose no data.
        center_on_mask: Center cropped cuboid at the mask.

    Raises:
        UserWarning: If your dataset does not contain files having the proper shape (240,240,155)

    Returns:
        __getitem__: Returns a dictionary containing:
            "gt_image": The cropped version of t1n.
            "voided_healthy_image": The cropped version of t1n where the healthy tissue is cropped away.
            "t1n_path": Path to the t1n file for this sample (relevant for reversing the sample).
            "healthy_mask": Cropped version of mask-healthy.
            "healthy_mask_path": Path to mask-healthy (relevant for reversing the sample)
            "cropped_bbox": Bounding box that was used for cropping
            "max_v": Maximal value of t1 image (used for normalization)
            "name": Sample ID. e.g. "BraTS-GLI-00003-000"
    """

    def __init__(self, root_dir: Path, crop_shape=(128, 128, 96), center_on_mask=False):
        # Initialize variables
        self.root_dir = root_dir
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask

        # Ground truth specific paths
        # multiline: list(root_dir.rglob("**/BraTS-GLI-*-*-t1n-????.nii.gz"))
        self.list_paths_t1n = list(root_dir.rglob("**/BraTS-GLI-*-*-t1n.nii.gz"))
        self.list_paths_mask_healthy = list(root_dir.rglob("**/BraTS-GLI-*-*-mask-healthy.nii.gz"))

    def __len__(self):
        return len(self.list_paths_mask_healthy)

    def preprocess(self, t1n: np.ndarray, healthy_mask: np.ndarray):
        """Transforms the images to a more unified format.

        Normalizes to -1,1. Crops to bounding box. Applies healthy mask to ground truth. Adds additional dimension in
        front for "channel".

        Args:
            t1n (np.ndarray): t1n from t1n file (ground truth).
            healthy_mask (np.ndarray): healthy mask from mask-healthy file.

        Raises:
            UserWarning: When your input images are not (240, 240, 155)

        Returns:
            t1n_voided_healthy_crop: Cropped version of voided t1n based on healthy tissue.
            t1n_max_v: Maximal value of t1n image (used for normalization).
            crop_box: Bounding box that was used for cropping.
            t1n_crop: The cropped version of t1n.
            healthy_mask_crop: Cropped version of the healthy mask.

        """

        # Size Assertions
        referenceShape = (240, 240, 155)
        if t1n.shape != referenceShape or healthy_mask.shape != referenceShape:
            raise UserWarning(f"Your t1n or healthy_mask shape is not {referenceShape}, they are: {t1n.shape} and {healthy_mask.shape}")

        # Normalize the image to [0,1]
        t1n[t1n < 0] = 0  # Values below 0 are considered to be noise.
        # Note that only 4 samples fulfill min(t1)!=0 : GLI-01332-000, GLI-00048-001, GLI-00446-000 and BraTS2023_01655
        t1n_max_v = np.max(t1n)
        t1n /= t1n_max_v

        # Crop to smaller size
        if self.center_on_mask:  # crop to region aground target
            shape = healthy_mask.shape[-3:]
            min_bbox = compute_bbox(healthy_mask)
            max_bbox = []
            for i, s in enumerate(min_bbox):
                s: slice
                d = self.crop_shape[i] - (s.stop - s.start)
                s_n = slice(s.start - d // 2, s.stop + ceil(d / 2))
                if s_n.start < 0:
                    s_n = slice(0, self.crop_shape[i])
                if s_n.stop > shape[i]:
                    s_n = slice(shape[i] - self.crop_shape[i], shape[i])

                max_bbox.append(s_n)
            max_bbox = tuple(max_bbox)
        else:  # crop to whole brain (removes everything empty space)
            max_bbox = compute_bbox(t1n)

        # apply crop
        t1n_crop = t1n[max_bbox]
        healthy_mask_crop = healthy_mask[max_bbox]

        # pad if too small
        t1n_crop, crop_box = pad3d(self.crop_shape, t1n_crop, max_bbox)
        healthy_mask_crop, _ = pad3d(self.crop_shape, healthy_mask_crop)

        # crop if bigger
        t1n_crop, healthy_mask_crop = random_crop(self.crop_shape, t1n_crop, healthy_mask_crop)

        # void ground truth where healthy
        t1n_voided_healthy_crop = t1n_crop * (1 - healthy_mask_crop)

        # map images to -1,1
        t1n_crop = normalize(t1n_crop)
        t1n_voided_healthy_crop = normalize(t1n_voided_healthy_crop)

        # Note: you might want do some data augmentation here (possible TODO)

        # add dimension for channel (we only have one though)
        t1n_crop = t1n_crop.unsqueeze(0)
        healthy_mask_crop = healthy_mask_crop.unsqueeze(0)
        t1n_voided_healthy_crop = t1n_voided_healthy_crop.unsqueeze(0)

        # set fitting datatype
        healthy_mask_crop = healthy_mask_crop.bool()

        return t1n_voided_healthy_crop, t1n_max_v, crop_box, t1n_crop, healthy_mask_crop

    def __getitem__(self, idx):
        # Ground truth: full t1n image
        t1n_path = self.list_paths_t1n[idx]
        t1n_img = nib.load(t1n_path)
        t1n = t1n_img.get_fdata()

        # Ground truth: mask (heathy mask) where we compute our loss on
        healthy_mask_path = self.list_paths_mask_healthy[idx]
        healthy_mask_img = nib.load(healthy_mask_path)
        healthy_mask = healthy_mask_img.get_fdata()

        # preprocess data
        t1n_voided_healthy_crop, t1n_max_v, crop_box, t1n_crop, healthy_mask_crop = self.preprocess(t1n, healthy_mask)

        # Output/Sample data
        sample_dict = {
            "gt_image": t1n_crop,
            "voided_healthy_image": t1n_voided_healthy_crop,
            "t1n_path": str(t1n_path),  # path to the t1n file for this sample.
            "healthy_mask": healthy_mask_crop,
            "healthy_mask_path": str(healthy_mask_path),
            "cropped_bbox": str(crop_box),  # bounding be used to crop
            # TODO: cropped_bbox CANNOT be given to dataloader/network directly (collate_fn -> type error)
            # currently we do str <-> eval conversion which is ugly.
            "max_v": t1n_max_v,  # maximal t1n_voided value used for normalization
            "name": healthy_mask_path.name[:19],
        }

        return sample_dict

    @classmethod
    def get_result_image(cls, prediction, sample):
        """Reverse the transformations that were necessary for the network.

        Args:
            prediction : numpy.ndarray
                The network output ( from model.forward() ).
            sample : dict
                The sample used as input for the network output.

        Returns:
            result : numpy.ndarray
                The resulting full scale inferred image
            img : nib.nifti1.Nifti1Image
                The same as "result" but as proper nifti file that can be saved to the file system
        """

        # Get target image
        t1n_img = nib.load(sample["t1n_path"])  # get target
        affine = t1n_img.affine
        t1n = t1n_img.get_fdata()

        # Get full healthy mask
        healthy_mask_img = nib.load(sample["healthy_mask_path"])
        healthy_mask = healthy_mask_img.get_fdata()
        voided_healthy_full = t1n * (1 - healthy_mask)

        # Remove channels
        prediction = prediction[0]

        # Reverse sample normalize
        prediction = (prediction + 1) / 2

        # reverse scaling
        prediction = prediction * sample["max_v"]

        # limit output to the inpainting region
        mask = sample["healthy_mask"][0]  # cropped mask, no channel
        prediction_minimal = np.zeros_like(prediction)
        prediction_minimal[mask] = prediction[mask]

        # reverse cropping
        bb = eval(sample["cropped_bbox"])
        prediction_full = np.zeros_like(voided_healthy_full)
        prediction_full[bb] = prediction_minimal

        # Place output in the inpainting regions
        result = voided_healthy_full + prediction_full

        # create output nifti image
        img = nib.nifti1.Nifti1Image(result, affine)

        return result, img
