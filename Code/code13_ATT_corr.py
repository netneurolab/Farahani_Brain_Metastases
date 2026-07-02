"""
###############################################################################

Compare with ATT map with MET maps

# Spearman's rank correlation

melanoma:
rho = 0.2958327453981795
p = 0.059940059940059943
q = 0.10989011

lung
rho = 0.5918844408245502
p = 0.001998001998001998
q = 0.00549451

breast
rho = 0.49890718483550023
p = 0.001998001998001998
q = 0.00549451

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
import nibabel as nib
from pathlib import Path
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from globals import path_medialwall, path_results
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Indicate type of cancer and load the medialwall data
#------------------------------------------------------------------------------

cancer_name = 'Breast'
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

#------------------------------------------------------------------------------
# Helper functionds
#------------------------------------------------------------------------------

def load_gifti_data(fp: Path) -> np.ndarray:
    """Load a GIFTI as a 1D float array of vertices."""
    img = nib.load(str(fp))
    data = img.agg_data()
    data = np.asarray(data).squeeze()
    if data.ndim != 1:
        data = np.asarray(data)[:, 0]
    return data.astype(float)

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

# ATT data
ATT_parc = np.load(path_results + 'parcellated_ATT.npy')*-1
ATT_parc = ATT_parc.flatten()

# Cancer data
cancer_cereb = np.load(path_results + cancer_name + '_34_cerebellum.npy')[:32]
cancer_ctx = np.load(path_results + cancer_name + '_400_surf.npy').flatten()

cancer_parc = np.concatenate((cancer_cereb, cancer_ctx))

#------------------------------------------------------------------------------
# Combined figure
#------------------------------------------------------------------------------

plt.figure(figsize=(5,5))
plt.scatter(ATT_parc, cancer_parc, s = 50, color = 'silver')
plt.scatter(ATT_parc[:32], cancer_parc[:32], s = 50, color = 'black')

# Regression line for visualization
coef = np.polyfit(ATT_parc, cancer_parc, 1)
x_line = np.linspace(np.nanmin(ATT_parc), np.nanmax(ATT_parc), 100)
y_line = coef[0] * x_line + coef[1]

plt.plot(x_line, y_line, color='black', linewidth=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_results + str(cancer_name) + '_ATT.svg', format = 'svg')
plt.show()

print(spearmanr(cancer_parc, ATT_parc))

#------------------------------------------------------------------------------
# Null correlation calculation
#------------------------------------------------------------------------------

nulls = np.load(path_results + 'nulls_' + cancer_name + '_10k_surf_brainspace.npy')[:,:1000]

r_nulls = np.zeros(1000)
for i in range(1000):
    r_nulls[i],_= spearmanr(nulls[i, :].flatten(), ATT_parc)
r_real,_ = spearmanr(ATT_parc, cancer_parc)

pval  = pval_cal(r_real, r_nulls, 1000)
print(pval)
plt.figure()
plt.hist(r_nulls)
plt.show()

p_spearmanr = [0.001998001998001998, 0.001998001998001998, 0.059940059940059943]
rej_b, q_b, _, _ = multipletests(p_spearmanr, alpha=0.05, method="fdr_by")
#--> array([ 0.00274725, -0.00549451,  0.05677656]) - with N = 1000; spearmanr and by

#------------------------------------------------------------------------------
# END