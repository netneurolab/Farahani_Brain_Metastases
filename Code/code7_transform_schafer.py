"""
###############################################################################

This is a code to create the schaefer atlas in the same dimention and affine
coordinate as that of MNI09a space.

The result is saved as a file names: 'schaefer_400-MNI09aSym.nii.gz'

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import subprocess
from globals import path_MNI09a, path_MNI06, path_data, path_results

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

in_img = path_data + 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'
mni06c = path_MNI06 + 'tpl-MNI152NLin6Asym_res-01_T1w.nii.gz'
mni09a = path_MNI09a + 'mni_icbm152_t1_tal_nlin_sym_09a.nii.gz'

output_dir = path_results + 'out_reg_schaefer/'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Output files
affine_mat = output_dir + 'affine_transform.mat'
warp_field = output_dir + 'nonlinear_warp.nii.gz'
warped_img = output_dir + 'schaefer_400-MNI09aSym.nii.gz' # this is what I finally need to get!
inverse_warp = output_dir + 'inverse_warp.nii.gz'

#------------------------------------------------------------------------------
#  Linear registration
#------------------------------------------------------------------------------

flirt_cmd = [
    'flirt',
    '-in', mni06c,
    '-ref', mni09a,
    '-out', output_dir + 'linear_registered.nii.gz',
    '-omat', affine_mat,
    '-bins', '256',
    '-cost', 'corratio',
    '-searchrx', '-90', '90',
    '-searchry', '-90', '90',
    '-searchrz', '-90', '90',
    '-dof', '12',
    '-interp', 'trilinear']

run_fsl_command(flirt_cmd, "Running FLIRT...")

#------------------------------------------------------------------------------
# Nonlinear registration
#------------------------------------------------------------------------------

fnirt_cmd = [
    'fnirt',
    '--in=' + mni06c,
    '--ref=' + mni09a,
    '--aff=' + affine_mat,   # Use affine from FLIRT as starting point
    '--cout=' + warp_field,  # Output warp field (coefficient field)
    '--iout=' + output_dir + 'fnirt_intermediate.nii.gz',  # Intermediate warped image
    '--verbose']

run_fsl_command(fnirt_cmd, "Running FNIRT...")

#------------------------------------------------------------------------------
# Apply warp to labels
#------------------------------------------------------------------------------

applywarp_cmd = [
    'applywarp',
    '--in=' + in_img,
    '--ref=' + mni09a,
    '--warp=' + warp_field,
    '--out=' + warped_img,
    '--interp=nn',
    '--datatype=int']

run_fsl_command(applywarp_cmd, "Applying warp to atlas labels...")

#------------------------------------------------------------------------------
# END