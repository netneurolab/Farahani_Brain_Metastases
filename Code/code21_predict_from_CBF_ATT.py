"""
###############################################################################
#------------------------------------------------------------------------------
Melanoma
#------------------------------------------------------------------------------

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.136
Model:                            OLS   Adj. R-squared:                  0.130
Method:                 Least Squares   F-statistic:                     22.46
Date:                Fri, 24 Apr 2026   Prob (F-statistic):           1.60e-13
Time:                        13:35:13   Log-Likelihood:                -581.40
No. Observations:                 432   AIC:                             1171.
Df Residuals:                     428   BIC:                             1187.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0215      0.046     -0.468      0.640      -0.112       0.069
x1             0.2244      0.047      4.745      0.000       0.131       0.317
x2             0.2414      0.047      5.098      0.000       0.148       0.335
x3             0.1452      0.062      2.334      0.020       0.023       0.268
==============================================================================
Omnibus:                       32.958   Durbin-Watson:                   1.631
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               40.014
Skew:                           0.644   Prob(JB):                     2.05e-09
Kurtosis:                       3.751   Cond. No.                         1.64
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Spin-based p-value for ATT: 0.1998001998001998
0.102
Spin-based p-value for CBF: 0.14585414585414586
0.07
Spin-based p-value for interaction: 0.2957042957042957
0.143

#------------------------------------------------------------------------------
Lung
#------------------------------------------------------------------------------

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.322
Model:                            OLS   Adj. R-squared:                  0.317
Method:                 Least Squares   F-statistic:                     67.73
Date:                Fri, 24 Apr 2026   Prob (F-statistic):           7.40e-36
Time:                        13:34:56   Log-Likelihood:                -529.07
No. Observations:                 432   AIC:                             1066.
Df Residuals:                     428   BIC:                             1082.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0010      0.041      0.023      0.981      -0.079       0.081
x1             0.5678      0.042     13.556      0.000       0.485       0.650
x2             0.0046      0.042      0.109      0.913      -0.078       0.087
x3            -0.0064      0.055     -0.117      0.907      -0.115       0.102
==============================================================================
Omnibus:                       37.215   Durbin-Watson:                   1.601
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               44.449
Skew:                           0.743   Prob(JB):                     2.23e-10
Kurtosis:                       3.513   Cond. No.                         1.64
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Spin-based p-value for ATT: 0.005994005994005994
0.003
Spin-based p-value for CBF: 0.965034965034965
0.513
Spin-based p-value for interaction: 0.958041958041958
0.526

#------------------------------------------------------------------------------
Breast
#------------------------------------------------------------------------------

                           OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.254
Model:                            OLS   Adj. R-squared:                  0.249
Method:                 Least Squares   F-statistic:                     48.62
Date:                Fri, 24 Apr 2026   Prob (F-statistic):           4.65e-27
Time:                        13:34:37   Log-Likelihood:                -549.63
No. Observations:                 432   AIC:                             1107.
Df Residuals:                     428   BIC:                             1124.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0231      0.043      0.542      0.588      -0.061       0.107
x1             0.5266      0.044     11.988      0.000       0.440       0.613
x2            -0.1616      0.044     -3.673      0.000      -0.248      -0.075
x3            -0.1562      0.058     -2.702      0.007      -0.270      -0.043
==============================================================================
Omnibus:                       69.212   Durbin-Watson:                   1.178
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              111.732
Skew:                           0.980   Prob(JB):                     5.47e-25
Kurtosis:                       4.539   Cond. No.                         1.64
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Spin-based p-value for ATT: 0.002997002997002997
0.001
Spin-based p-value for CBF: 0.2777222777222777
0.871
Spin-based p-value for interaction: 0.15084915084915085
0.918

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from functions import pval_cal
from globals import path_medialwall, path_results

#------------------------------------------------------------------------------
# Load medial wall
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

#------------------------------------------------------------------------------
# Helper functions
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

# Load cancer map
cancer_name = 'Breast'
cancer_parc_cereb = np.load(path_results + cancer_name+ '_34_cerebellum.npy')
cancer_parc_cortex = np.load(path_results + cancer_name + '_400_surf.npy').flatten()
cancer_parc = np.concatenate((cancer_parc_cereb[:32], cancer_parc_cortex), axis = 0)

# Load ATT
ATT_parc = np.load(path_results + 'parcellated_ATT.npy')*-1

# Load perfusion data
CBF_parc = np.load(path_results + 'parcellated_CBF.npy')

# Null brain maps
nulls =  np.load(path_results + 'nulls_' + cancer_name + '_10k_surf_brainspace.npy')[:1000, :].T

#------------------------------------------------------------------------------
# Build the linear model
#------------------------------------------------------------------------------

y = cancer_parc.astype(float).flatten()
att = ATT_parc.astype(float).flatten()
cbf = CBF_parc.astype(float).flatten()

m = np.isfinite(y) & np.isfinite(att) & np.isfinite(cbf)
y = y[m]; att = att[m]; cbf = cbf[m]

def z(x):
    return (x - np.mean(x)) / np.std(x)

y_z   = z(y)
att_z = z(att)
cbf_z = z(cbf)

interaction = att_z * cbf_z
X_int = sm.add_constant(np.column_stack([att_z, cbf_z, interaction]))
mdl_int = sm.OLS(y_z, X_int).fit()
print(mdl_int.summary())

nspins = 1000

beta1_null = np.zeros(nspins)
beta3_null = np.zeros(nspins)
beta2_null = np.zeros(nspins)

X_int = sm.add_constant(np.column_stack([att_z, cbf_z, interaction]))

for i in range(nspins):
    y_null = z(nulls[:, i])
    beta1_null[i] = sm.OLS(y_null, X_int).fit().params[1]
    beta2_null[i] = sm.OLS(y_null, X_int).fit().params[2]
    beta3_null[i] = sm.OLS(y_null, X_int).fit().params[3]

beta1_emp = mdl_int.params[1]
beta2_emp = mdl_int.params[2]
beta3_emp = mdl_int.params[3]

p_spin = pval_cal(beta1_emp, beta1_null,nspins)
print("Spin-based p-value for ATT:", p_spin)
print((np.sum(beta1_emp < beta1_null))/(nspins))

p_spin = pval_cal(beta2_emp, beta2_null,nspins)
print("Spin-based p-value for CBF:", p_spin)
print((np.sum(beta2_emp < beta2_null))/(nspins))

p_spin = pval_cal(beta3_emp, beta3_null,nspins)
print("Spin-based p-value for interaction:", p_spin)
print((np.sum(beta3_emp < beta3_null))/(nspins))

#------------------------------------------------------------------------------
# Plot null distributions
#------------------------------------------------------------------------------

betas_null = [beta1_null, beta2_null, beta3_null]
betas_emp  = [beta1_emp,  beta2_emp,  beta3_emp]
labels     = ["ATT", "CBF", "ATT × CBF"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for i, ax in enumerate(axes):
    null = betas_null[i]
    emp  = betas_emp[i]

    # Histogram of nulls
    ax.hist(null, bins=40, density=True, alpha=0.7, color = 'silver')

    # Empirical beta
    ax.axvline(emp, linestyle='--', linewidth=1, color = 'black')

    perc = (np.sum(null < emp) / len(null)) * 100
    ax.set_title(f"{labels[i]}\nemp={emp:.3f} | perc={perc:.1f}%", fontsize=11)
    ax.set_xlabel("β")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel("Density")
plt.suptitle(f"{cancer_name}: Null distributions of regression coefficients", y=1.05)
plt.savefig(path_results + cancer_name + '_beta_CBF_ATT_interaction.svg')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# END