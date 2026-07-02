"""
###############################################################################

This is a code to create a cerebellum atlas in the same dimention and affine
coordinate as that of MNI06 space.

The input file - cerebellum atlas - is: 'atl-Anatom_space-MNI09cSym_full_dseg.nii.gz'
The result is saved as a file names: 'atl-Anatom_space-MNI06_full_dseg.nii.gz'

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import subprocess
from globals import path_MNI06, path_MNI09c, path_diedrichsen, path_results

#------------------------------------------------------------------------------
# Function to run FSL commands
#------------------------------------------------------------------------------

def run_fsl_command(cmd, description):
    """Execute FSL command"""
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
        print("Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

#------------------------------------------------------------------------------
# Define paths and folders
#------------------------------------------------------------------------------

in_img = path_diedrichsen + 'atl-Anatom_space-MNI09cSym_full_dseg.nii.gz'

mni09c = path_MNI09c +  'mni_icbm152_t1_tal_nlin_sym_09c.nii.gz'
mni06 = path_MNI06 + 'tpl-MNI152NLin6Asym_res-01_T1w.nii.gz'

output_dir = path_results + 'out_reg_hcp_MNI06_ceb/'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Output files
affine_mat = output_dir + 'affine_transform.mat'
warp_field = output_dir + 'nonlinear_warp.nii.gz'
warped_img = output_dir + 'atl-Anatom_space-MNI06_full_dseg.nii.gz' # this is what I finally need to get!
inverse_warp = output_dir + 'inverse_warp.nii.gz'

#------------------------------------------------------------------------------
#  Linear registration
#------------------------------------------------------------------------------

flirt_cmd = [
    'flirt',
    '-in', mni09c,
    '-ref', mni06,
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
    '--in=' + mni09c,
    '--ref=' + mni06,
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
    '--ref=' + mni06,
    '--warp=' + warp_field,
    '--out=' + warped_img,
    '--interp=nn',
    '--datatype=int']

run_fsl_command(applywarp_cmd, "Applying warp to atlas labels...")

#------------------------------------------------------------------------------
# END