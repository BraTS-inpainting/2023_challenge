### Imports and Parameters ###

import shutil  # for: file copying
import numpy as np  # for: many data operations
import pandas as pd  # for: managing tabular data
import nibabel as nib  # for: managing nifti images
from scipy import ndimage  # for: connected component analysis and 3D image manipulation
from tqdm import tqdm  # for: nice progress bars
from pathlib import Path  # for: pythonic path representations
from multiprocessing import Pool  # for multi-processing


niftiShape = np.array([240, 240, 155])


### Binarize Existing Masks ###


# Takes original BraTS segmentation masks and returns processed binary segmentation mask.
def getBinarySegmentationMask(
    segmentationMask: np.ndarray, relevantLabels=[1, 2, 3], segmentationMinSize=800, fillHoles=True
) -> np.ndarray:
    """Takes original BraTS segmentation masks and returns processed binary segmentation mask.

    Has multiple options for processing the mask.

    Args:
        segmentationMask : numpy.ndarray, shape: 240, 240, 155
            Original BraTS tumor segmentation mask (from a _seg.nii.gz file). This mask has multiple labels.
        relevantLabels : list, optional, default=[1,2,3]
            List of original BraTS labels that will be considered for the "whole tumor" annotation. This variable
            relates to the labels in segmentationMask.
        segmentationMinSize : int, optional, default=800
            All binary segmentation compartments below this size (in voxels) will be not be considered/returned.
        fillHoles : bool, optional, default=True
            If True, fills holes in the 3D binary segmentation mask (if there are any), otherwise does nothing.

    Returns:
        binarySegmentationMask : numpy.ndarray, shape: 240, 240, 155
            A (binary) segmentation mask representing the (processed) whole tumor annotation.

    Side effects:
        None
    """

    # Create output structure
    binarySegmentationMask = np.zeros_like(segmentationMask, dtype=bool)

    # Add all relevant labels to binary mask
    for label in relevantLabels:
        binarySegmentationMask[segmentationMask == label] = True

    # Remove small connected components (presumably artifacts) from binary mask
    if segmentationMinSize > 0:
        connectivityKernel = np.ones((3, 3, 3))
        labeled, ncomponents = ndimage.label(binarySegmentationMask, connectivityKernel)  # get connected compartments
        for compLabel in range(1, ncomponents + 1):  # skip compartment with compartment label 0 as this is the background
            compartment_mask = labeled == compLabel
            if np.sum(compartment_mask) < segmentationMinSize:  # if compartment size is below threshold
                binarySegmentationMask[compartment_mask] = 0  # remove compartment from binary mask

    # Fill holes (if there are any)
    if fillHoles:
        binarySegmentationMask = ndimage.binary_fill_holes(binarySegmentationMask)

    return binarySegmentationMask


# Worker (Process) that generates the binary segmentation mask for one BraTS folder
def process_generateSegmentationMasks(packed_parameters: tuple) -> tuple:
    """Worker (Process) that generates the binary segmentation mask for one BraTS folder.

    Args:
        packed_parameters : tuple
            All variables the process needs to operate properly:
                folderName : str
                    Folder name of the respective folder in inputFolderRoot. Example: "BraTS-GLI-01337-000"
                inputFolderRoot : pathlib.Path
                    Path to reference BraTS dataset. Example:
                    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
                relevantLabels : list
                    List of original BraTS labels that will be considered for the "whole tumor" annotation. This variable
                    relates to the labels in segmentationMask.
                segmentationMinSize : list
                    All binary segmentation compartments below this size (in voxels) will be not be considered/returned.
                fillHoles : bool
                    If True, fills holes in the 3D binary segmentation mask (if there are any), otherwise does nothing.

    Returns:
        output data : tuple
            All data relating to the brain segmentation mask we later need:
                folderName : str
                    Same as input argument.
                brainMask : numpy.ndarray, shape: 1116000 ( <- (240*240*155)/8 )
                    The packed (see numpy.packbits()) brain mask. This brain masked is obtained from non-zero T1 voxels.
                brainMask_V: numpy.float64
                    Volume of brainMask measured in voxels.
                binarySegmentationMask : numpy.ndarray, shape: 240, 240, 155
                    The binary segmentation mask of the brain.
                p : numpy.float64
                    Volume of binarySegmentationMask measured in percentage of the brainMask.

    See also:
        getBinarySegmentationMask(): The function the converts a multi label image to a binary one
        generateBinarySegmentationMasks(): The function that creates these workers
        numpy.packbits(): The numpy function we use to efficiently store binary images


    Side effects:
        Read from filesystem

    """
    # unpack parameters
    folderName, inputFolderRoot, relevantLabels, segmentationMinSize, fillHoles = packed_parameters

    # Read in T1 image data
    #   btw. every file begins with its parent foldername. E.g in "BraTS-GLI-01337-000/" exists "BraTS-GLI-01337-000-seg.nii.gz"
    T1Path = inputFolderRoot.joinpath(folderName).joinpath(f"{folderName}-t1n.nii.gz")
    T1_flair = nib.load(T1Path)
    T1Data = T1_flair.get_fdata()

    # Read segmentation image data (multi label)
    segPath = inputFolderRoot.joinpath(folderName).joinpath(f"{folderName}-seg.nii.gz")
    img_seg = nib.load(segPath)
    segmentationMask = img_seg.get_fdata()

    ## Get brain mask ##
    brainMask = T1Data != 0  # non-zero T1 voxels

    # get compartments and their size
    labeled, ncomponents = ndimage.label(brainMask, np.ones((3, 3, 3)))  # gets connected compartments in 3D
    compartmentSizes = []
    for compLabel in range(1, ncomponents + 1):  # skip 0, which is background
        compartment_mask = labeled == compLabel
        compartment_size = np.sum(compartment_mask)
        compartmentSizes.append(compartment_size)

    sortIndices = np.argsort(compartmentSizes)[::-1]  # sort compartments by size and get their indices

    brainMask = labeled == sortIndices[0] + 1  # take biggest compartment ([0] is background -> take the next one)
    brainMask = ndimage.binary_fill_holes(brainMask)  # fill holes

    ## Get segmentation mask ##
    binarySegmentationMask = getBinarySegmentationMask(segmentationMask, relevantLabels, segmentationMinSize, fillHoles)
    brainMask_V = np.sum(brainMask)
    V = np.sum(binarySegmentationMask)  # amount of segmentation voxels
    p = V / brainMask_V  # volume of binary segmentation mask in percentage of brain

    # pack boolean arrays to reduce required space (traded against computation time)
    brainMask = np.packbits(brainMask)
    binarySegmentationMask = np.packbits(binarySegmentationMask)

    return (folderName, brainMask, brainMask_V, binarySegmentationMask, p)


# Takes original BraTS segmentation masks and returns a table of binary segmentation mask
def generateBinarySegmentationMasks(
    inputFolderRoot: Path, relevantLabels=[1, 2, 3], segmentationMinSize=800, fillHoles=True, forceRefresh=False, threads=16
) -> pd.DataFrame:
    """Takes original BraTS segmentation masks and returns a table of binary segmentation mask.

    The binary segmentation masks can be cleaned/post-processed (segmentationMinSize, fillHoles). Also, this function
    stores its results in the current working directory ("binarySegmentationMasks.gz", forceRefresh).

    Args:
        inputFolderRoot : pathlib.Path
            Path to reference BraTS dataset. Example:"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        relevantLabels : list, optional, default=[1,2,3]
            List of original BraTS labels that will be considered for the "whole tumor" annotation. This variable
            relates to the labels in segmentationMask.
        segmentationMinSize : int, optional, default=800
            All binary segmentation compartments below this size (in voxels) will be not be considered/returned.
        fillHoles : bool, optional, default=True
            If True, fills holes in the 3D binary segmentation mask (if there are any), otherwise does nothing.
        forceRefresh : bool, optional, default=False
            If True, (re)creates the "binarySegmentationMasks.gz" file, which stores the resulting binary segmentation masks.
            If False and "binarySegmentationMasks.gz" already exists, then the returned DataFrame is taken from the
            results file. If the results file does not exist, it is generated (as if forceRefresh was True).
        threads : int, optional, default=16
            The amount of parallel sub-processes to compute results.

    Returns:
        tumorSegmentations : pandas.DataFrame
            Table containing all generated binary segmentation masks. The DataFrame has the following columns:
                brainFolder : str, primary key.
                    Folder name of the respective folder in inputFolderRoot.
                brainMask : numpy.ndarray
                    The packed (see numpy.packbits()) brain mask. This brain masked is obtained from non-zero T1 voxels.
                brainMask_V : numpy.float64
                    Volume of brainMask measured in voxels
                tumorSegmentation : numpy.ndarray
                    The binary segmentation mask of the "whole tumor".
                tumorSegmentation_p : numpy.float64
                    Volume of binarySegmentationMask measured in percentage of the brainMask.

    Side effects:
        Read and write to filesystem

    See also:
        process_generateSegmentationMasks(): The actual worker process
        multiprocessing.Pool() : How to spawn sub-processes

    """

    # list all folders in the input directory (only foldernames)
    relevantFolders = sorted([x.stem for x in inputFolderRoot.glob("*")])

    # set path to results file
    filePath = Path("binarySegmentationMasks.gz")

    # if results file does not exist or we are forced to, the results are generated
    if forceRefresh or filePath.exists() == False:
        print(f"Processing {len(relevantFolders)} folders from {inputFolderRoot}.")

        # Create output structure that will later be converted to a DataFrame
        outputData = {
            "brainFolder": [],  # Folder name of the respective folder in inputFolderRoot
            "brainMask": [],  # mask of the "valid" brain (T1)
            "brainMask_V": [],  # volume of brainMask in voxels
            "tumorSegmentation": [],  # binary segmentation mask of tumor tissue
            "tumorSegmentation_p": [],  # volume of binary segmentation mask in percentage of brain
        }

        # Get/generate output data
        with tqdm(total=len(relevantFolders)) as pbar:  # progress bar
            for batch_index in range(0, len(relevantFolders), threads):  # work in batches of size=threads
                batch = relevantFolders[batch_index : batch_index + threads]

                # process input data
                threadData = []
                for folderName in batch:
                    threadData.append((folderName, inputFolderRoot, relevantLabels, segmentationMinSize, fillHoles))

                # execute processes in parallel
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_generateSegmentationMasks, threadData)

                # get process output data and append respective lists
                for dataSet in returnData:
                    folderName, brainMask, brainMask_V, tumorSegmentation, p = dataSet

                    outputData["brainFolder"].append(folderName)
                    outputData["brainMask"].append(brainMask)
                    outputData["brainMask_V"].append(brainMask_V)
                    outputData["tumorSegmentation"].append(tumorSegmentation)
                    outputData["tumorSegmentation_p"].append(p)

                pbar.update(len(batch))  # update progress bar status

        # convert and save output data
        print(f"Saving data to {filePath}")
        tumorSegmentations = pd.DataFrame(outputData)  # concert dict to DataFrame
        tumorSegmentations = tumorSegmentations.set_index("brainFolder", drop=True)  # use brainFolder as index / primary key
        tumorSegmentations.to_pickle(filePath)

    else:  # Read results from file
        print(f"Loading segmentation mask DataFrame from {filePath}")
        tumorSegmentations = pd.read_pickle(filePath)

    return tumorSegmentations


### Extract Isolated Tumor Segmentations ###


# Computes minimal bounding box of a 3D binary image
def getMinimalBB(data: np.ndarray) -> tuple:
    """Computes minimal bounding box of a 3D binary image.

    Args:
        data : numpy.ndarray
            The binary 3D image.

    Returns:
        limits : list
            A list of tuples (limits) for each dimension.
            Example for a 10x10x10 bounding box might be: ( (5,15), (7,17), (2,12) )
    """

    shape = data.shape
    limits = []  # where the limits in each dimension will go
    for axis in range(len(shape)):  # for each dimension get limits
        # get lower limit
        fr_indices = np.argmax(data, axis=axis)  # get first entry along current dimension
        validValuesMask = np.sum(data, axis=axis) > 0  # if there is no value we still obtain 0 as index
        fr_indices = fr_indices[validValuesMask]  # filter away the wrong 0 values
        fr = np.min(fr_indices)  # bound box should include the smallest value

        # get upper limit
        data_flipped = np.flip(data, axis=axis)  # view dimension from the other side
        to_indices = np.argmax(data_flipped, axis=axis)  # get first entry along dimension (viewed from other side)
        to_indices = to_indices[validValuesMask]  # filter away the wrong 0 values
        to_indices = shape[axis] - to_indices  # create actual upper limit by subtracting flipped indices from dimension length
        to = np.max(to_indices)  # bound box should include the maximal value

        # append limits for current axis
        limits.append((fr, to))

    return limits


# For one binary segmentation mask, get minimal bounding boxes of all components
def getMinimalBBs(tumorSegmentation: np.ndarray, connectivityKernel=np.ones((3, 3, 3))) -> list:
    """For one binary segmentation mask, get minimal bounding boxes of all components.

    Args:
        tumorSegmentation : numpy.ndarray
            The binary segmentation mask of the "whole tumor".
        connectivityKernel : numpy.ndarray of shape (3,3,3), optional, default=np.ones((3, 3, 3)).
            "A structuring element that defines feature connections." See scipy.ndimage.label()

    Returns:
        boxes : list of tuples
            Each tuple contains the bounding box limits for the respective dimension.

    Also see:
        scipy.ndimage.label(): The method we used to get connected components in 3D

    """
    labeled, ncomponents = ndimage.label(tumorSegmentation, connectivityKernel)  # get components

    # create bounding boxes for each connected component
    boxes = []
    # component label 0 is background: that bounding box is the shape of the input data
    for componentLabel in range(1, ncomponents + 1):
        bb = getMinimalBB(labeled == componentLabel)  # get BB for this component
        boxes.append(bb)

    return boxes


# Worker (Process) that extracts isolated segmentation components for one BraTS folder
def process_getTumorCompartments(packed_parameters: tuple) -> tuple:
    """Worker (Process) that extracts isolated segmentation components for one BraTS folder.

    Args:
        packed_parameters : tuple
            All variables the process needs to operate properly:
                folderName : str
                    Folder name of the respective folder in inputFolderRoot. Example: "BraTS-GLI-01337-000"
                brainMask_V: numpy.float64
                    Volume of brainMask measured in voxels.
                tumorSegmentation : numpy.ndarray, shape: 240, 240, 155
                    The binary segmentation mask of the "whole tumor".
    Returns:
        output data : tuple
            All data relating to the brain segmentation mask we later need. This is basically a list of columns.
            For each isolatable segmentation mask one row is generated.
                folderNames : list of str
                    Column of the folder name from which the segmentation component was extracted from. As one worker
                    processes one folder at a time, all entries in this column will be the same folderName from the
                    given as input.
                ps : list of numpy.float64
                    Column of the segmentation component size measured in percent of the brainMask.
                compartments : list of numpy.ndarray, variable shape
                    The segmentation compartment cropped from the whole brain (based on the minimal bounding box).
                    As the size of this crop can vary in size, this packed representation also varies in size.
                compartmentShapes : list of tuples
                    The shape of the segmentation compartment before it was packed (this is important for unpacking)

    See also:
        getTumorCompartments(): The function that creates these workers.
        getMinimalBBs(): How we get the bounding boxes for all segmentation compartments of a brain.
        numpy.packbits(): The numpy function we use to efficiently store binary images.

    Side effects:
        None

    """
    # extract parameters
    folderName, brainMask_V, tumorSegmentation = packed_parameters

    # setup output columns
    folderNames = []  # folder name
    ps = []  # mask volume percentage in relation to brain
    compartments = []  # (packed) tumor compartment
    compartmentShapes = []  # compartment shape

    for BB in getMinimalBBs(tumorSegmentation):  # for each BB = for each segmentation compartment
        folderNames.append(folderName)

        maskCrop = tumorSegmentation[BB[0][0] : BB[0][1], BB[1][0] : BB[1][1], BB[2][0] : BB[2][1]]  # crop compartment mask

        # size percentage
        shape = maskCrop.shape
        ps.append(np.sum(maskCrop) / brainMask_V)

        # stuff that is local (not brain globally) related to the masks
        compartments.append(np.packbits(maskCrop))
        compartmentShapes.append(shape)

    return folderNames, ps, compartments, compartmentShapes


# Get single tumor segmentation masks ##
def getTumorCompartments(tumorSegmentations: pd.DataFrame, forceRefresh=False, threads=16) -> pd.DataFrame:
    """Takes binary segmentation masks and extracts all isolated segmentation components.

    For each brain, this function finds connected components of tumor segmentation and returns a separate
    crop for each of these components (tumorCompartments).

    Args:
        tumorSegmentations : pandas.DataFrame
            Table containing all binary brain tumor segmentations. The DataFrame has the following columns:
                brainFolder : str, primary key.
                    Folder name of the respective folder in inputFolderRoot.
                brainMask : numpy.ndarray
                    The packed (see numpy.packbits()) brain mask. This brain masked is obtained from non-zero T1 voxels.
                brainMask_V : numpy.float64
                    Volume of brainMask measured in voxels
                tumorSegmentation : numpy.ndarray
                    The binary segmentation mask of the "whole tumor".
                tumorSegmentation_p : numpy.float64
                    Volume of binarySegmentationMask measured in percentage of the brainMask.
        forceRefresh : bool, optional, default=False
            If True, (re)creates the "tumorCompartments.gz" file, which stores the resulting tumor compartments.
            If False and "tumorCompartments.gz" already exists, then the returned DataFrame is taken from the
            results file. If the results file does not exist, it is generated (as if forceRefresh was True).
        threads : int, optional, default=16
            The amount of parallel sub-processes to compute results.

    Returns:
        tumorCompartments : pandas.DataFrame
            Table containing all extracted tumor segmentation compartments. The DataFrame has the following columns:
                brainFolder : str, primary key.
                    Folder name identifying the brain the compartment was taken from.
                p : numpy.float64
                    Column of the segmentation component size measured in percent of the brainMask.
                packedCompartment : numpy.ndarray, variable shape
                    The segmentation compartment cropped from the whole brain (based on the minimal bounding box).
                    As the size of this crop can vary in size, this packed representation also varies in size.
                compartmentShape : list of tuples
                    The shape of the segmentation compartment before it was packed (this is important for unpacking)

    Side effects:
        Read and write to filesystem

    See also:
        process_getTumorCompartments(): The actual worker process
        multiprocessing.Pool() : How to spawn sub-processes

    """

    # set path to results file
    filePath = Path("tumorCompartments.gz")

    # if results file does not exist or we are forced to, the results are generated
    if forceRefresh or filePath.exists() == False:
        # Create output structure that will later be converted to a DataFrame
        outputData = {
            "brainFolder": [],  # brainFolder where the masks were extracted from
            "p": [],  # volume of compartment in percentage of brain
            "packedCompartment": [],  # packed compartment
            "compartmentShape": [],  # shape of compartment before it was packed (important for un-packing)
        }

        # Get/generate output data
        indices = list(tumorSegmentations.index)
        with tqdm(total=len(indices)) as pbar:  # progress bar
            for batch_index in range(0, len(indices), threads):  # work in batches of size=threads
                batch = indices[batch_index : batch_index + threads]

                # process input data
                threadData = []
                for folderName in batch:
                    # unpack data
                    brainMask_V = tumorSegmentations["brainMask_V"][folderName]
                    tumorSegmentation = np.unpackbits(tumorSegmentations["tumorSegmentation"][folderName]).reshape(niftiShape).astype(bool)

                    # append data
                    threadData.append((folderName, brainMask_V, tumorSegmentation))

                # execute processes in parallel
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getTumorCompartments, threadData)

                # get process output data and append respective lists
                for dataSet in returnData:
                    folderNames, ps, compartments, compartmentShapes = dataSet

                    outputData["brainFolder"].extend(folderNames)
                    outputData["p"].extend(ps)
                    outputData["packedCompartment"].extend(compartments)
                    outputData["compartmentShape"].extend(compartmentShapes)

                pbar.update(len(batch))  # update progress bar status

        # convert and save output data
        print(f"Saving data to {filePath}")
        tumorCompartments = pd.DataFrame(outputData)
        tumorCompartments.to_pickle(filePath)

    else:  # Read results from file
        print(f"Loading segmentation compartments from {filePath}")

        tumorCompartments = pd.read_pickle(filePath)

    return tumorCompartments


### Generate Healthy Tissue Masks ###


# Get a semi-random location within the (positive) distance map, biased by the distance to the tumor
def sampleLocation(distanceMap: np.ndarray, minDistanceToTumor: float, rng: np.random._generator.Generator, randPointsN=2) -> tuple:
    """Get a semi-random location within the (positive) distance map, biased by the distance to the tumor.

    This function samples two (randPointsN) random points in the positive distance map (distanceMap) and returns the location
    that is further away from the tumor. The background and tumor voxels have distance 0 and are therefore no valid sampling
    target.

    Args:
        distanceMap : np.ndarray
            3D map of the brain where each voxel contains the minimal euclidean voxel distance to (dilated) brain tumor
            segmentation.
        minDistanceToTumor : float
            Minimal euclidean voxel distance to the tumor segmentation the sampled point must have.
        rng : numpy.random._generator.Generator
            Properly seeded random number generator from the subprocess calling this function.
        randPointsN : int, optional, default=2
            Amount of points to be sampled for choosing the best one. The more points are sampled the stronger is the
            tendency away from the tumor.

    Returns:
        point : tuple
            The sampled point in the brain.
    """
    # sample randPointsN random points inside the brain but not in the tumor
    validPoints = np.where(distanceMap > minDistanceToTumor)  # enforce minimal distance tu tumor
    randomIndices = np.round(rng.random(randPointsN) * len(validPoints[0])).astype(int)  # get random locations indices
    randomPoints = [(validPoints[0][index], validPoints[1][index], validPoints[2][index]) for index in randomIndices]

    # order points by euclidean voxel distance
    pointsDistance = [distanceMap[point] for point in randomPoints]
    sortIndices = np.argsort(pointsDistance)[::-1]
    orderedPoints = [randomPoints[index] for index in sortIndices]

    # take the location with highest distance
    point = orderedPoints[0]

    return point


# Semi-randomly samples a tumor segmentation compartment from the previously generated pool
def sampleCompartment(
    tumorCompartments: pd.DataFrame,
    sortedTumorCompartments: np.ndarray,
    tumorSegmentation_p: float,
    sizeRangeTolerance,
    rng: np.random._generator.Generator,
) -> np.ndarray:
    """Semi-randomly samples a tumor segmentation compartment from the previously generated pool.

    The sampling is only semi-random, as the size of the existing tumor segmentation in the target brain is already
    considered. More precisely, the compartment is chosen to be inversely proportional to the size of the existing
    tumor (Big tumor -> small compartment, and vice versa).
    Additionally, this function applies random transformations to the chosen segmentation compartment after it was chosen.
    This includes: random flipping/mirroring and random rotation in all 3 dimension.
    Code for random changes in size (zoom in, zoom out) exists but is commented out (include if you want to use that
    feature).

    Args:
        tumorCompartments : pandas.DataFrame
            Table containing all extracted tumor segmentation compartments. For more details see getTumorCompartments().
        sortedTumorCompartments : numpy.ndarray
            One dimensional array of all compartment sizes, sorted in ascending order. This represents the distribution
            of compartment sizes we orient ourself to choose inversely proportional for sampling.
        tumorSegmentation_p : numpy.float64
            Volume of brain tumor segmentation mask measured in percentage of the brainMask.
        sizeRangeTolerance : float
            Percentage of tolerance while choosing the inversely proportional size of the compartment. This parameter
            basically describes the width of the window in the size distribution we sample from. The location of this
            window results from the tumor size (tumorSegmentation_p).
        rng : np.random._generator.Generator
            Properly seeded random number generator from the subprocess calling this function.

    Returns:
        tumorCompartment : np.ndarray
            The unpacked tumor compartment which was sampled.

    """
    ## Select tumor mask ##
    brainIndex = np.searchsorted(sortedTumorCompartments, tumorSegmentation_p)  # which compartment would be as big as the existing tumor
    valuesCount = len(sortedTumorCompartments)  # how many compartments do we have in total
    targetIndex = valuesCount - brainIndex  # the inverse position on the size distribution
    fr = int(targetIndex - (sizeRangeTolerance * valuesCount) / 2)  # define lower window end
    if fr < 0:  # check that lower end is not blow 0
        fr = 0
    to = int(targetIndex + (sizeRangeTolerance * valuesCount) / 2)  # define upper window end
    if to >= valuesCount:  # check that upper end is not beyond the array limit
        to = valuesCount - 1
    tumorSegmentationCrop_index = int(fr + rng.random() * (to - fr))  # choose some compartment (index) within this window

    tumorCompartments_row = tumorCompartments.loc[[tumorSegmentationCrop_index]]  # get data for this compartment
    shape = tumorCompartments_row["compartmentShape"].item()  # get compartment shape (for unpacking)

    unpacked_length = shape[0] * shape[1] * shape[2]  # reconstruct how many boolean values the compartment has
    # Note: this is necessary, because the unpacked version might be slightly longer (rounded up to multiple of 8 bits/booleans)
    tumorCompartment = (
        np.unpackbits(tumorCompartments_row["packedCompartment"].item())[:unpacked_length].reshape(shape).astype(bool)
    )  # unpack

    ## Compartment transformations ##
    transformations = {}  # if you want to track the augmentations.
    # The transformations dict is currently not used for anything but you are free to modify the code and process it further

    # Transformation: Mirroring
    flipX, flipY, flipZ = np.sign(rng.random(3) * 2 - 1).astype(int)  # get 3 random flip values (1 or -1)
    tumorCompartment = tumorCompartment[::flipX, ::flipY, ::flipZ]  # flip respectively
    transformations["flip"] = [flipX, flipY, flipZ]  # note what flip we have applied (-1 for flipped, 1 for not flipped)

    # Transformation: Rotation
    rotXY, rotYZ = rng.random(2) * 360  # get two random angles between 0° and 360°
    tumorCompartment = ndimage.rotate(  # first rotation
        tumorCompartment,
        rotXY,
        axes=(0, 1),  # rotate in X-Y plane
        reshape=True,
        mode="grid-constant",  # works best for our case
        order=0,  # this is important! If interpolation would be carried out, we would get artifacts
        prefilter=False,  # should not matter as order is 0
    )
    tumorCompartment = ndimage.rotate(  # second rotation: in Y-Z plane
        tumorCompartment, rotYZ, axes=(1, 2), reshape=True, mode="grid-constant", order=0, prefilter=False
    )

    transformations["rotate"] = [rotXY, rotYZ]  # note which rotations (angles) were used

    # Transformation: Resize (currently not used, comment in if you want to)
    # sizeRange = (0.8,1.2) #+-20%
    # resizeFactor = sizeRange[0]+rng.random()*(sizeRange[1]-sizeRange[0]) # choose random resize factor in sizeRange
    # tumorCompartment = ndimage.zoom(tumorCompartment,resizeFactor,mode="grid-constant",order=0,prefilter=False) #do resize
    # transformations["resize"] = [resizeFactor] # note down resize factor

    return tumorCompartment


# Takes a center point and a compartment and computes where the bounding box in the brain would be.
def getTargetVolume(targetLocation: tuple, compartment) -> list:
    """Takes a center point and a compartment and computes where the bounding box in the brain would be.

    This function is necessary to map the isolated compartments - which are cropped from the brain - back into the
    brain reference frame. This "placement" of the compartment is of course depending on a target location.

    Args:
        targetLocation : tuple
            Target location where the compartment is placed. This point lies within the brain (not tumor, not background)
        compartment : numpy.ndarray
            The (unpacked) segmentation compartment that shall be placed at the targetLocation. For that, targetLocation
            is the geometric center of the compartment.

    Returns:
        targetVolume : list
            The bounding bounding box in the brain where the compartment will be.
    """

    compartment_shape = compartment.shape
    targetVolume = [
        (int(targetLocation[0] - np.floor(compartment_shape[0] / 2)), int(targetLocation[0] + np.ceil(compartment_shape[0] / 2))),
        (int(targetLocation[1] - np.floor(compartment_shape[1] / 2)), int(targetLocation[1] + np.ceil(compartment_shape[1] / 2))),
        (int(targetLocation[2] - np.floor(compartment_shape[2] / 2)), int(targetLocation[2] + np.ceil(compartment_shape[2] / 2))),
    ]
    return targetVolume


# Computes the minimal distance between two 3D binary masks.
def minDist(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the minimal euclidean distance between two 3D binary masks.

    Computes the minimal distance of each voxel to a and b each (distance maps). The sum of these distances in each
    point is the shortest path from a to b over this point. The overall shortest possible path from a to b is therefore
    the minimal value that exists in the summed up distance maps.

    Args:
        a : numpy.ndarray
            First binary mask.
        b : numpy.ndarray
            Second binary mask.

    Returns:
        min_dist : numpy.float64
            Minimal distance between shape a and b. If they overlap 0 is returned.
    """
    dist_a = ndimage.distance_transform_edt(~a)
    dist_b = ndimage.distance_transform_edt(~b)
    min_dist = np.min(dist_a + dist_b)
    return min_dist


# Check validity of compartment placement in the brain.
def validSampling(
    targetLocation: tuple,
    compartment: np.ndarray,
    tumorSegmentation: np.ndarray,
    brainMask: np.ndarray,
    minimalBrainIntersection_p: float,
    minDistanceToTumor,
):
    """Check validity of compartment placement in the brain.

    The location of the compartment at a given target location is checked for validity. If the combination is valid for
    this brain the return code 0 (returnData["ret"]) is returned. Otherwise the error code of the respective check.
    To be considered valid (ret = 0), a compartment has to:
     - be within the borders of the nifti cuboid, otherwise -> ret=1,2
     - not overlap with the (dilated) existing tumor, otherwise -> ret=3
     - overlap at to minimalBrainIntersection_p percent with the brain, otherwise -> ret=4
     - have a minimal euclidean distance to the (dilated) tumor of minDistanceToTumor voxels, otherwise -> ret=5

    Args:
        targetLocation : tuple
            The target location where the compartment should be placed.
        compartment : numpy.ndarray, various shapes
            The segmentation compartment cuboid.
        tumorSegmentation : numpy.ndarray, shape: 240, 240, 155
            The brain tumor segmentation mask of the target brain.
        brainMask : numpy.ndarray, shape: 240, 240, 155
            The (unpacked) brain mask. This brain masked is obtained from non-zero T1 voxels.
        minimalBrainIntersection_p : float
            How much percentage the compartment has to overlap at least with the brain (brainMask).
        minDistanceToTumor : float
            Minimal euclidean distance between the surfaces of the tumor and the compartment.

    Returns:
        ret : int
            The check return value. 0 if compartment placement is valid, otherwise error code.
        healthyMask : numpy.ndarray or None
            Either None (if checks failed first) or the healthy segmentation mask (has full nifti cuboid size).

    """

    # Check: Volume in range/borders?
    tV = getTargetVolume(targetLocation, compartment)
    if tV[0][0] < 0 or tV[1][0] < 0 or tV[2][0] < 0:  # check lower bound
        return 1, None
    if tV[0][1] > niftiShape[0] or tV[1][1] > niftiShape[1] or tV[2][1] > niftiShape[2]:  # check upper bound
        return 2, None

    # Check: Intersection with tumor mask
    if np.sum(np.logical_and(compartment, tumorSegmentation[tV[0][0] : tV[0][1], tV[1][0] : tV[1][1], tV[2][0] : tV[2][1]])) > 0:
        return 3, None

    # Check: minimal brain intersection
    brainIntersect_V = np.sum(np.logical_and(compartment, brainMask[tV[0][0] : tV[0][1], tV[1][0] : tV[1][1], tV[2][0] : tV[2][1]]))
    brainIntersect_p = brainIntersect_V / np.sum(compartment)
    if brainIntersect_p < minimalBrainIntersection_p:
        return 4, None

    # Check: minimal distance to Tumor (requires complete brain mask)
    healthyMask = np.zeros_like(brainMask)
    healthyMask[tV[0][0] : tV[0][1], tV[1][0] : tV[1][1], tV[2][0] : tV[2][1]] = compartment
    mask2mask_dist = minDist(healthyMask, tumorSegmentation)  # get minimal distance between surfaces
    if mask2mask_dist < minDistanceToTumor:
        return 5, healthyMask

    # Everything is fine
    return 0, healthyMask


# Worker (Process) that samples healthy segmentation masks from pool of existing tumor segmentation compartments.
def process_getHealthyMasks(packed_parameters):
    """Worker (Process) that samples healthy segmentation masks from pool of existing tumor segmentation compartments.

    Takes pool of tumor segmentation compartments, semi-random chooses on compartment, transforms it and places it
    semi-randomly in the brain. Potentially multiple resampling is required until the placement fulfills the given
    requirements.

    Args:
        packed_parameters : tuple
            Consists of general parameters (constant for every worker) and instance parameters (change for ever worker)
                generalParameters:
                    inputFolderRoot : pathlib.Path
                        Path to reference BraTS dataset.
                        Example:"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
                    outputFolderRoot : pathlib.Path
                        Path to the output folder. Example: "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
                    tumorCompartments : pandas.DataFrame
                        Table containing all extracted tumor segmentation compartments.
                        For mor details see getTumorCompartments()
                    sortedTumorCompartments : numpy.ndarray
                        One dimensional array of all compartment sizes, sorted in ascending order.
                    tumorDilationDistance : float
                        How much voxels the binary segmentation mask shall be dilated (inflated). This operation makes
                        the tumor segmentation bigger and also has a smoothing effect.
                    minDistanceToTumor : float
                        Minimal distance the sampled healthy tissue compartment needs to have to the (dilated) tumor
                        mask.
                    sizeRangeTolerance : float
                        Percentage of tolerance window for sampling inversely proportional in the compartment size
                        distribution. For more details see sampleCompartment()
                    randPointsN : int
                        Amount of points to be sampled for choosing the best one. The more points are sampled the
                        stronger is the tendency away from the tumor.
                    minimalBrainIntersection_p : float
                        How much percentage the compartment has to overlap at least with the brain (brainMask).
                    samplesPerBrain : int
                        How many healthy masks shall be sampled per brain. This parameter significantly influences the
                        size of the resulting training set.
                instanceParameters:
                    folderName : pathlib.Path
                        Folder name of the respective folder in inputFolderRoot. Example: "BraTS-GLI-01337-000"
                    tumorSegmentation : numpy.ndarray, shape: 240, 240, 155
                        The binary segmentation mask of the "whole tumor".
                    tumorSegmentation_p : numpy.float64
                        Volume of brain tumor segmentation mask measured in percentage of the brainMask.
                    brainMask : numpy.ndarray, shape: 240, 240, 155
                        The (unpacked) brain mask. Obtained from non-zero T1 voxels, used for reference what is valid
                        brain volume.
                    rng_seed : int
                        Semi-random seed for the random number generator of each worker.

    Returns:
        folderName : pathlib.Path
            Same as the folderName from instanceParameters from packed_parameters.
        healthyMasks : list of numpy.ndarray of shape: 1116000 ( <- (240*240*155)/8 )
            A list of packed (see numpy.packbits()) healthy tissue masks that were generated for this brain.
            The length of this list is equal to samplesPerBrain.

    See also:
        sampleLocation() : Semi-randomly samples a possible location for the healthy tissue mask.
        sampleCompartment() : Semi-randomly chooses a existing tumor segmentation compartment as healthy tissue mask.
        validSampling() : Checks whether the sampled configuration is valid given our criteria.
        getHealthyMasks() : The function that creates these workers.
        numpy.packbits() : Numpy function to efficiently store binary arrays.

    Side effects:
        Read and write on filesystem

    """

    # Unpack parameters
    generalParameters, instanceParameters = packed_parameters
    (  # general parameters
        inputFolderRoot,
        outputFolderRoot,
        tumorCompartments,
        sortedTumorCompartments,
        tumorDilationDistance,
        minDistanceToTumor,
        sizeRangeTolerance,
        randPointsN,
        minimalBrainIntersection_p,
        samplesPerBrain,
    ) = generalParameters
    # Instance specific parameters
    folderName, tumorSegmentation, tumorSegmentation_p, brainMask, rng_seed = instanceParameters

    # Create random number generator for this process
    rng = np.random.default_rng(seed=rng_seed)

    # Output data structure
    healthyMasks = []

    # Create distance map (relevant for mask placement) and dilate tumor#
    distanceMap = ndimage.distance_transform_edt(~tumorSegmentation)
    distanceMap -= tumorDilationDistance  # reducing the distance to the tumor effectively dilates the tumor
    tumorSegmentation[distanceMap < 0] = True  # update/dilate tumor segmentation mask

    distanceMap[brainMask == False] = 0  # set background as 0 (like tumor)

    # create healthy tissue masks for this brain
    for sample_id in range(samplesPerBrain):  # amount of mask per brain: samplesPerBrain
        # sample mask location based on distance to tumor
        targetLocation = sampleLocation(distanceMap, minDistanceToTumor, rng, randPointsN)

        # sample mask shape from existing tumor segmentation compartments (and do some transformations)
        tumorCompartment = sampleCompartment(tumorCompartments, sortedTumorCompartments, tumorSegmentation_p, sizeRangeTolerance, rng)

        # check if sampled configuration is valid
        ret, healthyMask = validSampling(
            targetLocation, tumorCompartment, tumorSegmentation, brainMask, minimalBrainIntersection_p, minDistanceToTumor
        )
        while ret > 0:  # resample until a proper configuration is found
            targetLocation = sampleLocation(distanceMap, minDistanceToTumor, rng, randPointsN)
            tumorCompartment = sampleCompartment(tumorCompartments, sortedTumorCompartments, tumorSegmentation_p, sizeRangeTolerance, rng)
            ret, healthyMask = validSampling(
                targetLocation, tumorCompartment, tumorSegmentation, brainMask, minimalBrainIntersection_p, minDistanceToTumor
            )

        # If a valid configuration was found: append to output list
        healthyMasks.append(healthyMask)

    # read original nifti file to get proper affine and header data
    segPath = inputFolderRoot.joinpath(folderName).joinpath(f"{folderName}-seg.nii.gz")
    img_seg = nib.load(segPath)

    # set and create output folder
    outputFolderName = folderName  # e.g. "BraTS-GLI-01337-000"
    outputFolderPath = outputFolderRoot.joinpath(outputFolderName)
    if not outputFolderPath.exists():  # make folder if it does not exists
        outputFolderPath.mkdir()

    # Create File: mask-unhealthy.nii.gz
    mask_unhealthy = np.zeros(niftiShape, dtype=bool)
    mask_unhealthy[tumorSegmentation] = True
    img = nib.Nifti1Image(mask_unhealthy, affine=img_seg.affine, header=img_seg.header)
    nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-mask-unhealthy.nii.gz"))

    # Create File: t1n.nii.gz
    T1Path = inputFolderRoot.joinpath(folderName).joinpath(f"{folderName}-t1n.nii.gz")
    shutil.copy(T1Path, outputFolderPath.joinpath(f"{outputFolderName}-t1n.nii.gz"))  # we just copy the original t1n

    T1_flair = nib.load(T1Path)
    T1Data = T1_flair.get_fdata()

    # Create sample specific files: inference mask, t1-voided and healthy tissue mask
    for i, healthyMask in enumerate(healthyMasks):
        # Create File: mask-healthy.nii.gz / mask-healthy-{i:04d}.nii.g
        mask_healthy = np.zeros(niftiShape, dtype=bool)
        mask_healthy[healthyMask] = True
        img = nib.Nifti1Image(mask_healthy, affine=img_seg.affine, header=img_seg.header)
        if samplesPerBrain > 1:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-mask-healthy-{i:04d}.nii.gz"))
        else:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-mask-healthy.nii.gz"))

        # Create File: mask.nii.gz / mask-{i:04d}.nii.gz
        mask = np.zeros(niftiShape, dtype=bool)
        mask[healthyMask] = True
        mask[tumorSegmentation] = True
        img = nib.Nifti1Image(mask, affine=img_seg.affine, header=img_seg.header)
        if samplesPerBrain > 1:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-mask-{i:04d}.nii.gz"))
        else:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-mask.nii.gz"))

        # Create File: t1n-voided.nii.gz / t1n-voided-{i:04d}.nii.gz
        t1n_voided = T1Data.copy()
        t1n_voided[mask == True] = 0
        img = nib.Nifti1Image(t1n_voided, affine=img_seg.affine, header=img_seg.header)
        if samplesPerBrain > 1:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-t1n-voided-{i:04d}.nii.gz"))
        else:
            nib.save(img, outputFolderPath.joinpath(f"{outputFolderName}-t1n-voided.nii.gz"))

    # compress healthy masks before returning them
    healthyMasks = [np.packbits(healthyMask) for healthyMask in healthyMasks]

    return folderName, healthyMasks


# Randomly creates healthy masks based on existing segmentation masks and some heuristic
def getHealthyMasks(
    inputFolderRoot,
    outputFolderRoot,
    tumorSegmentations,
    tumorCompartments,
    samplesPerBrain=1,
    tumorDilationDistance=5.0,
    minDistanceToTumor=5.0,
    sizeRangeTolerance=0.1,
    randPointsN=2,
    minimalBrainIntersection_p=0.75,
    forceRefresh=False,
    threads=16,
    seed=2023,
):
    """Samples healthy segmentation masks from pool of existing tumor segmentation compartments.

    Takes pool of tumor segmentation compartments, semi-random chooses on compartment, transforms it and places it
    semi-randomly in the brain. Potentially multiple resampling is required until the placement fulfills the given
    requirements.

    Args:
        inputFolderRoot : pathlib.Path
            Path to reference BraTS dataset.
            Example:"ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        outputFolderRoot : pathlib.Path
            Path to the output folder. Example: "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
        tumorSegmentations : pandas.DataFrame
            Table containing all generated binary segmentation masks.
            For more details see generateBinarySegmentationMasks()
        tumorCompartments : pandas.DataFrame
            Table containing all extracted tumor segmentation compartments.
            For mor details see getTumorCompartments()
        samplesPerBrain : int, optional, default=1
            How many healthy masks shall be sampled per brain. This parameter significantly influences the size of the
            resulting training set.
        tumorDilationDistance : float, optional, default=5.0
            How much voxels the binary segmentation mask shall be dilated (inflated). This operation makes the tumor
            segmentation bigger and also has a smoothing effect.
        minDistanceToTumor : float, optional, default=5.0
            Minimal distance the sampled healthy tissue compartment needs to have to the (dilated) tumor mask.
        sizeRangeTolerance : float, optional, default=0.1
            Percentage of tolerance window for sampling inversely proportional in the compartment size
            distribution. For more details see sampleCompartment()
        randPointsN : int, optional, default=0.1
            Amount of points to be sampled for choosing the best one. The more points are sampled the
            stronger is the tendency away from the tumor.
        minimalBrainIntersection_p : float, optional, default=0.75
            How much percentage the compartment has to overlap at least with the brain (brainMask).
        forceRefresh : bool, optional, default=False
            If True, (re)creates the samples in outputFolderRoot and "healthyMasks.gz", which stores the resulting
            healthy masks too. If False and "healthyMasks.gz" already exists, then the returned DataFrame is taken
            from the results file. No samples will be (re)generated. If the output folder or the results file does
            not exist, both are generated as if forceRefresh was True.
        threads : int, optional, default=16
            The amount of parallel sub-processes to compute results.
        seed : int, optional, default=2023
            Seed to make the sampling process reproducible.

    Returns:
        healthyMasks : pandas.DataFrame
            Table containing all extracted tumor segmentation compartments. The DataFrame has the following columns:
                brainFolder : str, primary key.
                    Folder name identifying the brain the healthy mask(s) was/were generated for.
                healthyMasks : list ofnumpy.ndarray of shape: 1116000 ( <- (240*240*155)/8 )
                    A list of packed (see numpy.packbits()) healthy tissue masks that were generated for the respective
                    brain. The length of this list is equal to samplesPerBrain.

    See also:
        sampleLocation() : Semi-randomly samples a possible location for the healthy tissue mask.
        sampleCompartment() : Semi-randomly chooses a existing tumor segmentation compartment as healthy tissue mask.
        validSampling() : Checks whether the sampled configuration is valid given our criteria.
        getHealthyMasks() : The function that creates these workers.
        numpy.packbits() : Numpy function to efficiently store binary arrays.

    Side effects:
        Read and write on filesystem

    """

    # set path to results file
    filePath = Path("healthyMasks.gz")

    # if results file does not exist or we are forced to, the results are generated
    if forceRefresh or filePath.exists() == False or outputFolderRoot.exists() == False:
        # Create output structures
        outputData = {"brainFolder": [], "packedMasks": []}  # folder name of output folder/brain  # list of list of packed healthy masks
        outputFolderRoot.mkdir(exist_ok=True)

        # Initialize random number generator with our seed to obtain reproducible sampling
        np.random.seed(seed)

        # Sort pool of tumor compartment by size
        sizeSortedIndices = np.argsort(tumorCompartments.p)
        sortedTumorCompartments = np.array(tumorCompartments.p[sizeSortedIndices])

        # Parameters that stay the same for each brain
        generalParams = (
            inputFolderRoot,
            outputFolderRoot,
            tumorCompartments,
            sortedTumorCompartments,
            tumorDilationDistance,
            minDistanceToTumor,
            sizeRangeTolerance,
            randPointsN,
            minimalBrainIntersection_p,
            samplesPerBrain,
        )

        # Get/generate output data
        indices = list(tumorSegmentations.index)
        with tqdm(total=len(indices)) as pbar:  # progress bar
            for batch_index in range(0, len(indices), threads):  # work in batches of size=threads
                batch = indices[batch_index : batch_index + threads]

                # create input data
                threadData = []
                for folderName in batch:
                    # parameters specific for this brain
                    instanceParameters = (
                        folderName,
                        np.unpackbits(tumorSegmentations["tumorSegmentation"][folderName])
                        .reshape(niftiShape)
                        .astype(bool),  # tumorSegmentation
                        tumorSegmentations["tumorSegmentation_p"][folderName],  # tumorSegmentation_p
                        np.unpackbits(tumorSegmentations["brainMask"][folderName]).reshape(niftiShape).astype(bool),  # brainMask
                        int(np.random.rand() * 10**14),  # rng_seed
                    )

                    # append data
                    threadData.append((generalParams, instanceParameters))

                # execute processes in parallel
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getHealthyMasks, threadData)

                # get process output data and append respective lists
                for dataSet in returnData:
                    folderName, healthyMasks = dataSet

                    outputData["brainFolder"].append(folderName)
                    outputData["packedMasks"].append(healthyMasks)

                pbar.update(len(batch))  # update progress bar status

        # convert and save output data
        print(f"Saving data to {filePath}")
        healthyMasks = pd.DataFrame(outputData)
        healthyMasks = healthyMasks.set_index("brainFolder", drop=True)  # use brainFolder as index
        healthyMasks.to_pickle(filePath)

    else:  # Read results from file
        print(f"Loading healthy masks from {filePath}")
        healthyMasks = pd.read_pickle(filePath)

    return healthyMasks
