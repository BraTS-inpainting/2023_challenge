# The Dataset
The [dataset for training](https://www.synapse.org/#!Synapse:syn51522870) can be downloaded on the official Synapse website after [registering for the challenge](https://www.synapse.org/#!Synapse:syn51156910/wiki/622347).

## Dataset structure
The ```BraTS2023_Dataset_Local_Synthesis``` folder contains one sub-folder for each brain.
Sub-folders are named ```BraTS-GLI-XXXXX-YYY``` with *XXXXX* being a 5 digit integer (with leading zeros) and YYY either 000 or 001. For example: *BraTS-GLI-01337-000*. Within each sub-folder, the following files are given:
- For inference, the following files are provided to the network:
  - ```BraTS-GLI-XXXXX-YYY-mask.nii.gz```: mask (binary) specifying which voxels shall be in-painted
  - ```BraTS-GLI-XXXXX-YYY-t1n-voided.nii.gz```: T1 scan where the to-be-infilled section is 0. 
- For the training phase of the challenge, the following files will **additionally** be provided:
  - ```BraTS-GLI-XXXXX-YYY-mask-healthy.nii.gz```: mask (binary) specifying an exemplary tumor mask in healthy tissue (inference target)
  - ```BraTS-GLI-XXXXX-YYY-mask-unhealthy.nii.gz```: mask (binary) specifying the (slightly enlarged) tumor annotation region (inference target)
  - ```BraTS-GLI-XXXXX-YYY-t1n.nii.gz```: full T1 scan. This scan therefore includes the **ground truth** for the voided T1 image which is to be infilled.


## Dataset generation (optional)
It is also possible to generate the ```BraTS2023_Dataset_Local_Synthesis``` dataset locally. The Jupyter Notebook ```dataset_generation.ipynb``` contains a step-by-step guide to generate our dataset. Example pictures as well as some documentation is also provided there. Note that our dataset is based on the general BraTS2023-GLI dataset. Therefore, to generate our dataset you first need to [download](https://www.synapse.org/#!Synapse:syn51514105) the general BraTS2023-GLI dataset.

### Requirements
- A python3 environment containing the following packages: ```numpy pandas nibabel scipy tqdm matplotlib jupyter```




