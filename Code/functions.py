import os
import globals
import  scipy.io
import numpy as np
import nibabel as nib
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.images import load_data, dlabel_to_gifti
from globals import path_templates, path_medialwall, path_atlas

#%%
def save_as_dscalar_and_npy(input_data_voxelwise, type_data, path_save, name_save):
    """
        Save the input data as both a CIFTI dscalar file (.dscalar.nii) for visualization and a NumPy (.npy) file.

        This function takes voxel-wise input data, reshapes it, and saves it in two formats:
        1. A CIFTI dscalar (.dscalar.nii) file, which is useful for visualization in tools like Connectome Workbench.
        2. A NumPy (.npy) file, which stores the data in a raw format for further analysis.

        Args:
            input_data_voxelwise (numpy array): The voxel-wise input data. The data should be a 1D.
            type_data (str): Specifies the type of data being saved. Options are:
                             - 'cortex': Data corresponds to cortical vertices (59k).
                             - 'cortex_subcortex': Data corresponds to both cortical and subcortical vertices/voxels (92k).
            path_save (str): The directory where the output files will be saved.
            name_save (str): The base name for the saved files (without the extension).

        Returns:
            None. The function saves the data in two formats:
                - A .dscalar.nii file for visualization.
                - A .npy file for raw data storage.

        Notes:
            - The function uses predefined templates for 'cortex' and 'cortex_subcortex'.
             (Ensure that the correct template files are available at the specified paths.)
            - The input data will be reshaped to match the expected format of the dscalar file.

        Example Usage:
            - To save cortical data:
                save_as_dscalar_and_npy(cortex_data, 'cortex', path_to_save, 'cortex_output')

            - To save combined cortical and subcortical data:
                save_as_dscalar_and_npy(cortex_subcortex_data, 'cortex_subcortex', path_to_save, 'cortex_subcortex_output')
    """
    # Load the templates for 'cortex' and 'cortex_subcortex'
    template_paths = {
        'cortex': os.path.join(path_templates, 'cortex.dscalar.nii'),
        'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
    templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}

    # Choose the correct template based on the 'type_data' argument
    if type_data == 'cortex':
        template = templates['cortex']
    if type_data == 'cortex_subcortex':
        template = templates['cortex_subcortex']

    # Save the input data as a CIFTI dscalar (.dscalar.nii) file
    new_img = nib.Cifti2Image(input_data_voxelwise.reshape(1, -1),
                              header = template.header,
                              nifti_header = template.nifti_header)
    new_img.to_filename(os.path.join(path_save, name_save + '.dscalar.nii'))

    # Save the input data as a NumPy (.npy) file
    np.save(path_save + name_save + '.npy', input_data_voxelwise)
#%%
def save_gifti(file, file_name):
    """
    Generate as a func.gii gile - for visualization in workbench
    """
    da = nib.gifti.GiftiDataArray(file, datatype = 'NIFTI_TYPE_FLOAT32')
    img = nib.GiftiImage(darrays = [da])
    nib.save(img, (file_name +'.func.gii'))
#%%
def pval_cal(rho_actual, null_dis, num_spins):
    """
    Calculate p-value - non parametric
    """
    p_value = (1 + np.count_nonzero(abs((null_dis - np.mean(null_dis))) > abs((rho_actual - np.mean(null_dis))))) / (num_spins + 1)
    return(p_value)
#%%#%%
def convert_cifti_to_parcellated_SchaeferTian(data, type_data, version, path_save, name_save):
    """
        Convert vertex-wise data to a parcellated format using the Schaefer-Tian atlas (versions S1 or S4).

        This function converts vertex-wise data from CIFTI files to parcellated data based on the Schaefer-Tian atlas. 
        The function supports parcellation of the cortex, subcortex, or both, and adjusts the number of parcels depending 
        on the version specified ('S1' or 'S4'). For cortical data, the version parameter is ignored.

        Args:
            data (numpy array): The vertex-wise data to be parcellated. The data array should have dimensions of 
                                ( number of timepoints, number of vertices/voxels).
            type_data (str): Specifies which data should be parcellated. Options are:
                             - 'cortex': Only cortical data will be parcellated.
                             - 'cortex_subcortex': Both cortical and subcortical data will be parcellated.
                             - 'subcortex': Only subcortical data will be parcellated.
                              -'subcortex_double':  Only subcortical data will be parcellated (n | n+x)
            version (str): Specifies the version of the Schaefer-Tian atlas to use. Options are:
                           - 'S1': Uses 16 subcortical parcels.
                           - 'S4': Uses 54 subcortical parcels.
            path_save (str): The directory where the output file will be saved.
            name_save (str): The base name for the saved file.

        Returns:
            numpy array: The parcellated data. The shape depends on the `type_data` and `version` arguments. For example:
                         - If 'cortex', the output will have shape (400, number of timepoints).
                         - If 'cortex_subcortex', the output will have shape (400 + 16/54, number of timepoints).
                         - If 'subcortex', the output will have shape (16/54, number of timepoints).
                           If 'subcortex_double', the output will have shape ((8/27), number of timepoints).

        Notes:
            - The function uses the Schaefer-Tian atlas for parcellating the data.
            - Ensure that the input data matches the expected shape for the chosen `type_data`.
            - The output is also saved as a NumPy array in a .npy format.

        Example Usage:
            - To convert cortical data using version 'S1':
                parcellated_data = convert_cifti_to_parcellated_SchaeferTian(data, 'cortex', 'S1', path_to_save, 'cortex_data')

            - To convert both cortical and subcortical data using version 'S4':
                parcellated_data = convert_cifti_to_parcellated_SchaeferTian(data, 'cortex_subcortex', 'S4', path_to_save, 'full_data')
        """
    if type_data == 'cortex':
        # For cortical data, ignore version
        version = None  # Set version to None for cortex to ensure it doesn't affect naming
    else:
        # Set the number of nodes for Schaefer-Tian atlas based on the version
        nnodes_Schaefer_version = globals.nnodes_Schaefer_S1 if version == 'S1' else globals.nnodes_Schaefer_S4
        nnodes_version = globals.nnodes_S1 if version == 'S1' else globals.nnodes_S4
        yeo_tian = nib.load(path_atlas + f'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_{version}.dlabel.nii').get_fdata()
        yeo_tian_subcortex = yeo_tian[0, globals.num_cort_vertices_noMW:]

    # Load the medial wall mask and the Schaefer-Tian atlas for the selected version
    mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

    # Fetch the cortical atlas from the Schaefer parcellation
    schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
    atlas = load_data(dlabel_to_gifti(schaefer))
    yeo_tian_cortical = atlas[(mask_medial_wall.flatten()) == 1]

    # Split the data into cortex and subcortex components
    data_cortex = data[:, :globals.num_cort_vertices_noMW]
    if type_data in ['cortex_subcortex', 'subcortex', 'subcortex_double']:
        data_subcortex = data[:, globals.num_cort_vertices_noMW:]

    if type_data == 'cortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((globals.nnodes_Schaefer, np.shape(data)[0]))
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_cortex[:, yeo_tian_cortical == n], axis = 1)

    if type_data == 'cortex_subcortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((nnodes_Schaefer_version, np.shape(data)[0]))
        for n in range(1, nnodes_version + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, yeo_tian_subcortex == n], axis = 1)
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_parcellated[n - 1 + nnodes_version,:] = np.nanmean(data_cortex[:, yeo_tian_cortical == n], axis = 1)

    if type_data == 'subcortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((nnodes_version, np.shape(data)[0]))
        for n in range(1, nnodes_version + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, yeo_tian_subcortex == n], axis = 1)

    if type_data == 'subcortex_double':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((int(nnodes_version/2), np.shape(data)[0]))
        for n in range(1, int(nnodes_version/2) + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, (yeo_tian_subcortex == n) | (yeo_tian_subcortex == n + int(nnodes_version/2))], axis = 1)

    # Save the parcellated data array as a NumPy file and also return it
    file_name_suffix = f"_{type_data}_parcellated" + (f"_{version}" if version else "")
    np.save(os.path.join(path_save, f"{name_save}{file_name_suffix}.npy"), data_parcellated)
    return data_parcellated
#%%
def save_parcellated_data_in_SchaeferTian_forVis(data, type_data, version, path_save, name_save):
    """
    Save a parcel-wise map based on the Schaefer-Tian atlas for visualization in workbench.

    This function converts parcel-wise data based on the Schaefer-Tian atlas into a CIFTI dscalar format, suitable for 
    visualization in Connectome Workbench. The function supports data for the cortex, subcortex, or both.

    Args:
        data (numpy array): The parcel-wise data to be saved. The expected shape depends on the `type_data` argument:
                            - 'cortex': Data should have shape (400, 1) corresponding to the 400 cortical parcels.
                            - 'cortex_subcortex': Data should have shape (400 + 16/54, 1), where the additional 
                              parcels correspond to the subcortex depending on the version.
                            - 'subcortex': Data should have shape (16/54, 1), depending on the version.
        type_data (str): Specifies which type of data is being processed. Options are:
                         - 'cortex': Process and save only cortical data.
                         - 'cortex_subcortex': Process and save both cortical and subcortical data.
                         - 'subcortex': Process and save only subcortical data.
        version (str): Specifies the version of the atlas being used. Options are:
                       - 'S1': Version with 16 subcortical parcels.
                       - 'S4': Version with 54 subcortical parcels.
        path_save (str): The path where the output file will be saved.
        name_save (str): The base name of the saved file.

    Returns:
        None. The function saves the parcel-wise data in a .dscalar.nii file format, which can be visualized using 
        Connectome Workbench.

    Notes:
        - The number of parcels in the subcortex depends on the version ('S1' or 'S4'). 'S1' uses 16 parcels, while 'S4' uses 54 parcels.
        - Ensure that the data passed as input matches the expected shape for the specified `type_data`.

    Example Usage:
        - To save data for the cortex only:
            save_parcellated_data_in_SchaeferTian_forVis(cortex_data, 'cortex', 'S1', path_to_save, 'cortex_data')

        - To save data for both cortex and subcortex:
            save_parcellated_data_in_SchaeferTian_forVis(full_data, 'cortex_subcortex', 'S4', path_to_save, 'full_data')
    """
    if type_data == 'cortex':
        # For cortical data, ignore version and use the standard Schaefer parcellation
        version = None  # Set version to None for cortex to ensure it doesn't affect naming
    else:
        # Set the number of nodes for Schaefer-Tian atlas based on the version
        nnodes_version = globals.nnodes_S1 if version == 'S1' else globals.nnodes_S4
        yeo_tian = nib.load(path_atlas + f'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_{version}.dlabel.nii').get_fdata()

        yeo_tian_subcortex = yeo_tian[0, globals.num_cort_vertices_noMW:]


    # Load the medial wall mask and the Schaefer-Tian atlas for the appropriate version
    mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

    # Fetch the cortical atlas from the Schaefer parcellation
    schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
    atlas = load_data(dlabel_to_gifti(schaefer))
    yeo_tian_cortical = atlas[(mask_medial_wall.flatten()) == 1]

    # Initialize the data array to save
    data_to_save = np.zeros(globals.num_vertices_voxels)

    # Process cortical data
    if type_data == 'cortex':
        data_to_save_cortex = np.zeros(globals.num_cort_vertices_noMW)
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_to_save_cortex[yeo_tian_cortical == n] = data[n - 1]
        data_to_save[:globals.num_cort_vertices_noMW] = data_to_save_cortex

    # Process both cortical and subcortical data
    if type_data == 'cortex_subcortex':
        data_to_save_cortex = np.zeros(globals.num_cort_vertices_noMW)
        data_to_save_subcortex = np.zeros(globals.num_subcort_voxels)
        for n in range(1, nnodes_version + 1):
            data_to_save_subcortex[yeo_tian_subcortex == n] = data[n - 1]
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_to_save_cortex[yeo_tian_cortical == n] = data[n - 1 + nnodes_version]
        data_to_save[globals.num_cort_vertices_noMW:] = data_to_save_subcortex
        data_to_save[:globals.num_cort_vertices_noMW] = data_to_save_cortex

    # Process subcortical data only
    if type_data == 'subcortex':
        data_to_save_subcortex = np.zeros(globals.num_subcort_voxels)
        for n in range(1, nnodes_version + 1):
            data_to_save_subcortex[yeo_tian_subcortex == n] = data[n - 1]
        data_to_save[globals.num_cort_vertices_noMW:] = data_to_save_subcortex
    # Load the appropriate template for saving
    template_paths = {'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
    templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
    template = templates['cortex_subcortex']
    # Create and save the new CIFTI image
    new_img = nib.Cifti2Image(data_to_save.reshape(1, -1),
                              header=template.header,
                              nifti_header=template.nifti_header)

    # Save map
    file_name_suffix = f"_{type_data}_parcellated" + (f"_{version}" if version else "")
    new_img.to_filename(os.path.join(path_save, f"{name_save}{file_name_suffix}" +'.dscalar.nii'))
#%%
def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()
#%%
def calculate_mean_per_parcel(data_voxel, atlas, n_parcels):
    """
    Calculate mean w-score across parcels.
    """
    parcelwise_data = np.zeros((int(n_parcels), 1))
    for i in range(1, int(n_parcels) + 1):
        parcelwise_data[i - 1, :] = np.nanmean(data_voxel[atlas == i])
    return parcelwise_data
#%%