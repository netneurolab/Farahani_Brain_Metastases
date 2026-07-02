"""
###############################################################################

Similarity of brain maps (3 brain met maps) - N-perm = 1000

    Lung vs Breast: r=0.5021, p=0.000999, q=0.005495, sig=True
    Lung vs Melanoma: r=0.1298, p=0.354645, q=0.650183, sig=False
    Breast vs Melanoma: r=0.3310, p=0.009990, q=0.027473, sig=True

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from functions import pval_cal
import matplotlib.pyplot as plt
from globals import path_results
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Load cancer data
#------------------------------------------------------------------------------

lun_cancer_cereb = np.load(path_results + 'Lung_34_cerebellum.npy')[:32]
lun_cancer_ctx = np.load(path_results + 'Lung_400_surf.npy').flatten()
lung_cancer_parc = np.concatenate((lun_cancer_cereb, lun_cancer_ctx))

breast_cancer_cereb = np.load(path_results + 'Breast_34_cerebellum.npy')[:32]
breast_cancer_ctx = np.load(path_results + 'Breast_400_surf.npy').flatten()
breast_cancer_parc = np.concatenate((breast_cancer_cereb, breast_cancer_ctx))

mel_cancer_cereb = np.load(path_results + 'Melanoma_34_cerebellum.npy')[:32]
mel_cancer_ctx = np.load(path_results + 'Melanoma_400_surf.npy').flatten()
mel_cancer_parc = np.concatenate((mel_cancer_cereb, mel_cancer_ctx))

#------------------------------------------------------------------------------
# Load null maps
#------------------------------------------------------------------------------

breast_nulls = np.load(path_results + 'nulls_Breast_10k_surf_brainspace.npy').T
lung_nulls = np.load(path_results + 'nulls_Lung_10k_surf_brainspace.npy').T
mel_nulls = np.load(path_results + 'nulls_Melanoma_10k_surf_brainspace.npy').T

#------------------------------------------------------------------------------
# Correlate maps
#------------------------------------------------------------------------------

print('lung and breast')
print(spearmanr(breast_cancer_parc, lung_cancer_parc))

print('lung and melanoma')
print(spearmanr(mel_cancer_parc, lung_cancer_parc))

print('melanoma and breast')
print(spearmanr(breast_cancer_parc, mel_cancer_parc))

#------------------------------------------------------------------------------
# Assess similarity of the maps
#------------------------------------------------------------------------------

def pairwise_similarity_test(map1, map2, map1_nulls, name1, name2, nspins=1000):
    """
    map1, map2:        real full maps, shape (432,)
    map1_nulls:        nulls for map1, shape (432, nspins)
    """
    # Real similarity
    r_real, _ = spearmanr(map1, map2)

    # Null distribution
    r_null = np.zeros(nspins)
    for i in range(nspins):
        r_null[i], _ = spearmanr(map1_nulls[:, i], map2)

    # p-value
    p = pval_cal(r_real, r_null, nspins)

    print(f'{name1} vs {name2}')
    print(f'Observed Spearman r = {r_real:.4f}')
    print(f'p_spin = {p:.6f}')
    print('-----------------------------------')
    return r_real, p, r_null

r_lb, p_lb, null_lb = pairwise_similarity_test(
    lung_cancer_parc, breast_cancer_parc, lung_nulls, 'Lung', 'Breast')

r_lm, p_lm, null_lm = pairwise_similarity_test(
    lung_cancer_parc, mel_cancer_parc, mel_nulls, 'Lung', 'Melanoma')

r_bm, p_bm, null_bm = pairwise_similarity_test(
    breast_cancer_parc, mel_cancer_parc, mel_nulls, 'Breast', 'Melanoma')

#------------------------------------------------------------------------------
# Multiple-testing correction
#------------------------------------------------------------------------------

pair_names = [
    ('Lung', 'Breast'),
    ('Lung', 'Melanoma'),
    ('Breast', 'Melanoma')]

p_vals = np.array([p_lb, p_bm, p_lm], dtype=float)
r_vals = np.array([r_lb, r_bm, r_lm], dtype=float)

reject_fdr, p_vals_fdr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_by')

print('Raw p-values:', p_vals)
print('FDR-corrected q-values:', p_vals_fdr)
print('Significant after FDR:', reject_fdr)

for (name1, name2), r, p, q, sig in zip(pair_names, r_vals, p_vals, p_vals_fdr, reject_fdr):
    print(f'{name1} vs {name2}: r={r:.4f}, p={p:.6f}, q={q:.6f}, sig={sig}')

#------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------

labels = ['Breast','Lung', 'Melanoma']
n_maps = len(labels)

r_mat = np.eye(n_maps)
p_mat = np.zeros((n_maps, n_maps))
q_mat = np.zeros((n_maps, n_maps))

# Fill symmetric entries
r_mat[0, 1] = r_mat[1, 0] = r_lb
r_mat[0, 2] = r_mat[2, 0] = r_bm 
r_mat[1, 2] = r_mat[2, 1] = r_lm

p_mat[0, 1] = p_mat[1, 0] = p_lb
p_mat[0, 2] = p_mat[2, 0] = p_bm
p_mat[1, 2] = p_mat[2, 1] = p_lm
np.fill_diagonal(p_mat, np.nan)

q_mat[0, 1] = q_mat[1, 0] = p_vals_fdr[0]
q_mat[0, 2] = q_mat[2, 0] = p_vals_fdr[1]
q_mat[1, 2] = q_mat[2, 1] = p_vals_fdr[2]
np.fill_diagonal(q_mat, np.nan)

annot_mat = np.empty((n_maps, n_maps), dtype=object)
for i in range(n_maps):
    for j in range(n_maps):
        if i == j:
            annot_mat[i, j] = '1.00'
        else:
            q = q_mat[i, j]
            stars = ''
            if q < 0.001:
                stars = '***'
            elif q < 0.01:
                stars = '**'
            elif q < 0.05:
                stars = '*'
            annot_mat[i, j] = f'r={r_mat[i,j]:.2f}\nq={q:.3g}{stars}'

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(r_mat, vmin=-1, vmax=1, cmap='coolwarm')
ax.set_xticks(np.arange(n_maps))
ax.set_yticks(np.arange(n_maps))
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(labels, fontsize=12)
for i in range(n_maps):
    for j in range(n_maps):
        ax.text(j, i, annot_mat[i, j],
                ha='center', va='center', fontsize=10, color='black')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Spearman r', rotation=90)
ax.set_title('Similarity of brain metastasis maps', fontsize=13)
plt.tight_layout()
plt.savefig(path_results + 'similarity_of_brain_met.svg')
plt.show()

#------------------------------------------------------------------------------
# END