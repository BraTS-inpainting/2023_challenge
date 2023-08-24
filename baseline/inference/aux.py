import nibabel as nib


def write_nifti(input_array, reference_nifti_path, output_nifti_path):
    reference = nib.load(reference_nifti_path)

    the_nifti = nib.Nifti1Image(input_array, reference.affine, reference.header)
    nib.save(the_nifti, output_nifti_path)


def read_nifti(input_nifti_path):
    the_nifti = nib.load(input_nifti_path)
    return the_nifti.get_fdata()


def inpaint_numpy_array(
    voided_input,
    inpainting_mask,
):
    # TODO load model weights and run prediction
    predicted = "TODO"
    return predicted


def infer_single_case(
    voided_nifti_file,
    inpainting_mask_nifti_file,
    prediction_nifti_file,
):
    voided = read_nifti(voided_nifti_file)
    mask = read_nifti(inpainting_mask_nifti_file)

    inpainted = inpaint_numpy_array(
        voided_input=voided,
        inpainting_mask=mask,
    )

    write_nifti(
        input_array=inpainted,
        output_nifti_path=prediction_nifti_file,
        reference_nifti_path=voided_nifti_file,
    )
    return True
