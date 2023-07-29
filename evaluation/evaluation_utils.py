from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
import torch
import numpy as np

# Define evaluation Metrics
psnr = PeakSignalNoiseRatio(data_range=1.0) #because we normalize to 0-1
ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
mse = MeanSquaredError()


def __percentile_clip(input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
    """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
    Percentiles for normalization can come from another tensor.

    Args:
        input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
            If reference_tensor is None, the percentiles from this tensor will be used.
        reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
        p_min (float, optional): Lower end percentile. Defaults to 0.5.
        p_max (float, optional): Upper end percentile. Defaults to 99.5.
        strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

    Returns:
        torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
    """
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

    return output_tensor

            

def compute_metrics(gt_image: torch.Tensor, prediction: torch.Tensor, mask: torch.Tensor, normalize=True):
    """Computes MSE, PSNR and SSIM between two images only in the masked region.

    Normalizes the two images to [0;1] based on the gt_image 0.5 and 99.5 percentile in the non-masked region.
    Requires input to have shape (1,1, X,Y,Z), meaning only one sample and one channel.
    For MSE and PSNR we use the respective torchmetrics libraries on the voxels that are covered by the mask.
    For SSIM, we first zero all non-mask voxels, then we apply regular SSIM on the complete volume. In the end we take
    the "full SSIM" image from torchmetrics and only take the values relating to voxels within the mask.
    The main difference between the original torchmetrics SSIM and this substitude for masked images is that we pad
    with zeros while torchmetrics does reflection padding at the cuboid borders.
    This does slightly bias the SSIM voxel values at the mask surface but does not influence the resulting participant
    ranking as all submission underlie the same bias.

    Args:
        gt_image (torch.Tensor): The t1n ground truth image (t1n.nii.gz)
        prediction (torch.Tensor): The inferred/predicted t1n image
        mask (torch.Tensor): The inference mask (mask.nii.gz)
        normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
            region. Defaults to True

    Raises:
        UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

    Returns:
        float: (MSE, PSNR, SSIM)
    """

    if not (prediction.shape[0] == 1 and prediction.shape[1] == 1):
        raise UserWarning(f"All inputs have to be 5D with the first two dimensions being 1. Your prediction dimension: {prediction.shape}")

    # Get Infill region (we really are only interested in the infill region)
    prediction_infill = prediction * mask
    gt_image_infill = gt_image * mask

    # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
    if normalize:
        reference_tensor = gt_image * ~mask #use all the tissue that is not masked for normalization
        gt_image_infill = __percentile_clip(gt_image_infill, reference_tensor=reference_tensor, p_min=0.5, p_max=99.5, strictlyPositive=True)
        prediction_infill = __percentile_clip(prediction_infill, reference_tensor=reference_tensor, p_min=0.5, p_max=99.5, strictlyPositive=True)


    # SSIM - apply on complete masked image but only take values from masked region
    full_cuboid_SSIM, ssim_idx_full_image = ssim(preds=prediction_infill, target=gt_image_infill)
    ssim_idx = ssim_idx_full_image[mask]
    SSIM = ssim_idx.mean()

    # only voxels that are to be inferred (-> flat array)
    gt_image_infill = gt_image_infill[mask]
    prediction_infill = prediction_infill[mask]

    # MSE
    MSE = mse(preds=prediction_infill, target=gt_image_infill)

    # PSNR
    PSNR = psnr(preds=prediction_infill, target=gt_image_infill)

    return float(MSE), float(PSNR), float(SSIM)
