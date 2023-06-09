from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError
import torch

# Define evaluation Metrics
psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure(return_full_image=True)
mse = MeanSquaredError()


def compute_metrics(gt_image: torch.Tensor, output: torch.Tensor, mask: torch.Tensor, normalize=True):
    """Computes MSE, PSNR and SSIM between two images only in the masked region.

    Normalizes the two images to [0;1] based on the gt_image maximal value in the masked region.
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
        output (torch.Tensor): The inferred t1n image
        mask (torch.Tensor): The inference mask (mask.nii.gz)
        normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
            region. Defaults to True

    Raises:
        UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

    Returns:
        _type_: MSE, PSNR, SSIM as float each
    """

    if not (output.shape[0] == 1 and output.shape[1] == 1):
        raise UserWarning(f"All inputs have to be 5D with the first two dimensions being 1. Your output dimension: {output.shape}")

    # Get Infill region (we really are only interested in the infill region)
    output_infill = output * mask
    gt_image_infill = gt_image * mask

    # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
    if normalize:
        v_max = gt_image_infill.max()
        output_infill /= v_max
        gt_image_infill /= v_max

    # SSIM - apply on complete masked image but only take values from masked region
    full_cuboid_SSIM, ssim_idx_full_image = ssim(gt_image_infill, output_infill)
    ssim_idx = ssim_idx_full_image[mask]
    SSIM = ssim_idx.mean()

    # only voxels that are to be inferred (-> flat array)
    gt_image_infill = gt_image_infill[mask]
    output_infill = output_infill[mask]

    # MSE
    MSE = mse(gt_image_infill, output_infill)

    # PSNR - similar to pytorch PeakSignalNoiseRatio until 4 digits after decimal point
    PSNR = 10.0 * torch.log10((torch.max(gt_image_infill) - torch.min(gt_image_infill)) ** 2 / MSE)

    return float(MSE), float(PSNR), float(SSIM)
