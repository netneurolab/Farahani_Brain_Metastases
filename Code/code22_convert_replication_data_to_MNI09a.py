"""
###############################################################################

To tranfer replication data to MNI09a

    This is a code to create a sth in the same dimention and affine
    coordinate as that of MNI09s space

    The input file is:  'Lung_distribution_HEAD.nii.gz' / 'Breast_distribution_HEAD.nii.gz'
    The result is saved as a file names: Lung_distribution_HEAD-MNI09aSym.nii.gz / Breast_distribution_HEAD-MNI09aSym.nii.gz

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import subprocess
import nibabel as nib
from globals import path_MNI09a, path_MNI09c, path_results, path_validation

#------------------------------------------------------------------------------
# Helper function to run FSL commands
#------------------------------------------------------------------------------

def run_fsl_command(cmd, description):
    """Execute FSL command and check for errors"""
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("Success.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

#------------------------------------------------------------------------------
# Define paths and folders
#------------------------------------------------------------------------------

img = nib.load(path_validation + 'Breast/Breast_distribution_Breast+tlrc.HEAD')
nib.save(img, path_validation + 'Breast_distribution_HEAD.nii')

in_img = path_validation + 'Breast_distribution_HEAD.nii'

mni09c = path_MNI09c + 'mni_icbm152_t1_tal_nlin_sym_09c.nii.gz'
mni09a = path_MNI09a + 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'

output_dir = path_results + 'out_reg_ceb/'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Output files
affine_mat = output_dir + 'affine_transform.mat'
warp_field = output_dir + 'nonlinear_warp.nii.gz'
warped_img = output_dir + 'Breast_distribution_HEAD-MNI09aSym.nii.gz' # this is what I finally want to get!
inverse_warp = output_dir + 'inverse_warp.nii.gz'

#------------------------------------------------------------------------------
# Apply warp
#------------------------------------------------------------------------------

applywarp_cmd = [
    'applywarp',
    '--in=' + in_img,
    '--ref=' + mni09a,
    '--warp=' + warp_field,
    '--out=' + warped_img,
    '--interp=nn']

run_fsl_command(applywarp_cmd, "Applying warp field to atlas labels...")

#------------------------------------------------------------------------------
# END