"""
###############################################################################

Spearman's rank correlation - with surf method - N = 1000 - by

Breast_node_neighbour DIST  r=-0.5259856918910198  p=0.2087912087912088
Breast_node_neighbour FC  r=0.4806683944517974  p=0.01098901098901099
Breast_node_neighbour ATT  r=0.5811926539958455  p=0.001998001998001998
Breast_node_neighbour CBF  r=0.5094001156222855  p=0.000999000999000999
Breast_node_neighbour GENE  r=0.49486814475052804  p=0.014985014985014986

-----------------------

Lung_node_neighbour DIST  r=-0.5368818420023256  p=0.37962037962037964
Lung_node_neighbour FC  r=0.4490083096819673  p=0.16383616383616384
Lung_node_neighbour ATT  r=0.6114613418496112  p=0.001998001998001998
Lung_node_neighbour CBF  r=0.5920786826918439  p=0.002997002997002997
Lung_node_neighbour GENE  r=0.5260751473410149  p=0.016983016983016984

-----------------------

Melanoma_node_neighbour DIST  r=-0.3943980645472423  p=0.44255744255744256
Melanoma_node_neighbour FC  r=0.3346835003664548  p=0.2597402597402597
Melanoma_node_neighbour ATT  r=0.29499564481453106  p=0.22777222777222778
Melanoma_node_neighbour CBF  r=0.2503234387568044  p=0.16183816183816183
Melanoma_node_neighbour GENE  r=0.3658348351727517  p=0.18181818181818182

-----------------------

q-matrix (columns Breast/Lung/Mel; rows DIST/FC/ATT/CBF/GENE):
     0.94475051 1.         1.         
     0.10939216 0.90497154 0.99447422
     0.03314914 0.03314914 0.94475051
     0.03314914 0.03729278 0.90497154
     0.12075758 0.12075758 0.90497154

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import seaborn as sns
import nibabel as nib
import scipy.ndimage as ndi
from functions import pval_cal
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from globals import path_results, path_FC
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Settings and helper functions
#------------------------------------------------------------------------------

n_nodes = 432
nspins = 1000
eps = 1e-6

use_inverse_distance = False

def prep_W(W, zero_diag=True, drop_neg=True):
    W = W.copy()
    if zero_diag:
        np.fill_diagonal(W, 0)
    if drop_neg:
       # W[W < 0] = 0
       W = np.abs(W)
    return W

def node_to_neighbour(map_vec, W):
    denom = np.nansum(W, axis=1)
    denom = np.where(denom == 0, np.nan, denom)
    return (W @ map_vec) / denom

def compute_r_and_nulls(map_vec, nulls, W):
    neigh_real = node_to_neighbour(map_vec, W)
    r_real, _ = spearmanr(map_vec, neigh_real)

    r_spins = np.zeros(nspins)
    for n in range(nspins):
        null_map = nulls[:, n]
        null_neigh = node_to_neighbour(null_map, W)
        r_spins[n], _ = spearmanr(null_map, null_neigh)

    p_spin = pval_cal(r_real, r_spins.reshape(-1, 1), nspins)
    return r_real, p_spin, r_spins, neigh_real

#------------------------------------------------------------------------------
# Load distance matrix
#------------------------------------------------------------------------------

atlas_path = os.path.join(path_results, 'cerebellum_cortex_atlas_09.nii.gz')
atlas = nib.load(atlas_path).get_fdata()
atlas = atlas - 2
atlas[atlas<=0] = 0

com = np.zeros((n_nodes, 3))
c = 0
for lab in np.unique(atlas)[1:]:
    temp = (atlas == lab)
    com[c, :] = ndi.center_of_mass(temp)
    c += 1

distance = cdist(com, com)  # (432, 432)

# Distance weights
if use_inverse_distance:
    distW = 1.0 / (distance + eps)
else:
    distW = distance.copy()

distW = prep_W(distW, zero_diag=True, drop_neg=False) # distance isn't negative anyway

#------------------------------------------------------------------------------
# Load similarity matrices (ATT, CBF, Gene)
#------------------------------------------------------------------------------

gene_unmasked = np.load(path_results + 'gene_similarity_432.npy')
gene_unmasked[gene_unmasked<0] = 0

per = np.load(path_results + 'zscored_blood_similarity_432.npy')
per = prep_W(per, zero_diag=True, drop_neg=True)

att = np.load(path_results + 'zscored_blood_arrival_similarity_432.npy')
att = prep_W(att, zero_diag=True, drop_neg=True)

gene = gene_unmasked
gene = prep_W(gene, zero_diag=True, drop_neg=True)

#------------------------------------------------------------------------------
# FC and its settings
#------------------------------------------------------------------------------

fc_use_abs = False     # False -> "+ only" (drop negatives)
                       # True -> abs(FC)
fc_drop_neg = True     # keep True if fc_use_abs=False

subject_files = sorted([
    f for f in os.listdir(path_FC)
    if f.endswith('.npy')])
num_subjects = len(subject_files)

FC_all = np.zeros((432, 432, num_subjects))
for s in range(num_subjects):
    FC_all[:,:, s] = np.load(path_FC + subject_files[s])

FC = np.mean(FC_all, axis=2)                           # (432,432)

FC = prep_W(FC, zero_diag=True, drop_neg=False)        # diag=0, keep sign for now
if fc_use_abs:
    FC = np.abs(FC)
else:
    if fc_drop_neg:
        FC[FC < 0] = 0                                 # "+ only" like your earlier snippet

np.save(path_results + 'FC_average.npy', FC)

#------------------------------------------------------------------------------
# Similarity matrix visualization: 5 heatmaps side-by-side
#------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(27, 6))

# 1) DIST
sns.heatmap(distance, vmin=-1*np.nanmax(distance), vmax=np.nanmax(distance), cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[0])
axes[0].set_title('Distance (cdist)')
axes[0].axhline(y=32, color='black', linewidth=0.8)
axes[0].axvline(x=32, color='black', linewidth=0.8)

# 2) FC - this is only +
sns.heatmap(FC, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[1])
axes[1].set_title('FC similarity')
axes[1].axhline(y=32, color='black', linewidth=0.8)
axes[1].axvline(x=32, color='black', linewidth=0.8)

# 3) ATT
sns.heatmap(att, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[2])
axes[2].set_title('ATT similarity')
axes[2].axhline(y=32, color='black', linewidth=0.8)
axes[2].axvline(x=32, color='black', linewidth=0.8)

# 4) CBF
sns.heatmap(per, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[3])
axes[3].set_title('CBF similarity')
axes[3].axhline(y=32, color='black', linewidth=0.8)
axes[3].axvline(x=32, color='black', linewidth=0.8)

# 5) GENE
sns.heatmap(gene, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[4])
axes[4].set_title('Gene similarity')
axes[4].axhline(y=32, color='black', linewidth=0.8)
axes[4].axvline(x=32, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(path_results + "Heatmaps_DIST_FC_ATT_CBF_GENE.png", dpi=300)
plt.show()

#------------------------------------------------------------------------------
# Load cancer data and nulls
#------------------------------------------------------------------------------

breast_cancer_cereb = np.load(path_results + 'Breast_34_cerebellum.npy')
breast_cancer_ctx = np.load(path_results + 'Breast_400_surf.npy').flatten()
breast_data = np.concatenate((breast_cancer_cereb[:32], breast_cancer_ctx), axis = 0)
breast_nulls = np.load(path_results + 'nulls_Breast_10k_surf_brainspace.npy').T
breast_nulls = breast_nulls[:,:1000]

lung_cancer_cereb = np.load(path_results + 'Lung_34_cerebellum.npy')
lung_cancer_ctx = np.load(path_results +  'Lung_400_surf.npy').flatten()
lung_data = np.concatenate((lung_cancer_cereb[:32], lung_cancer_ctx), axis = 0)
lung_nulls = np.load(path_results +'nulls_Lung_10k_surf_brainspace.npy').T
lung_nulls = lung_nulls[:,:1000]

melanoma_cancer_cereb = np.load(path_results + 'Melanoma_34_cerebellum.npy')
melanoma_cancer_ctx = np.load(path_results +  'Melanoma_400_surf.npy').flatten()
melanoma_data = np.concatenate((melanoma_cancer_cereb[:32], melanoma_cancer_ctx), axis = 0)
melanoma_nulls = np.load(path_results + 'nulls_Melanoma_10k_surf_brainspace.npy').T
melanoma_nulls = melanoma_nulls[:,:1000]

#------------------------------------------------------------------------------
# Node - neighbour (same structure, now 4 columns: ATT, CBF, GENE, DIST)
#------------------------------------------------------------------------------

def build_neighbour_matrix(map_vec):
    neigh = np.zeros((n_nodes, 5))
    for i in range(n_nodes):
        neigh[i, 0] = np.nansum(map_vec * distW[i, :]) / np.nansum(distW[i, :])  # DIST
        neigh[i, 1] = np.nansum(map_vec * FC[i, :])    / np.nansum(FC[i, :])     # FC
        neigh[i, 2] = np.nansum(map_vec * att[i, :])   / np.nansum(att[i, :])    # ATT
        neigh[i, 3] = np.nansum(map_vec * per[i, :])   / np.nansum(per[i, :])    # CBF
        neigh[i, 4] = np.nansum(map_vec * gene[i, :])  / np.nansum(gene[i, :])   # GENE
    return neigh

breast_neighbour_abnormality = build_neighbour_matrix(breast_data)
lung_neighbour_abnormality   = build_neighbour_matrix(lung_data)
melanoma_neighbour_abnormality = build_neighbour_matrix(melanoma_data)

#------------------------------------------------------------------------------
# Main analysis function: compute r + spin p-value
#------------------------------------------------------------------------------

def main_function(cancer_parc, neighbour_abnormality, nulls, title_cancer):
    # Real r
    val_corr = np.zeros(5)
    pvals  = [None]*5
    rnulls = [None]*5

    names = ["DIST", "FC", "ATT", "CBF", "GENE"]
    Ws    = [distW,   FC,   att,   per,   gene]

    for k in range(5):
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(x=neighbour_abnormality[:, k], y=cancer_parc,
                    color='silver',
                    scatter_kws={'s': 80, 'edgecolor': 'black', 'linewidth': 0.8})
        ax.set_title(f"{title_cancer} | {names[k]}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path_results + f"{title_cancer}_{names[k]}.svg", format="svg")
        plt.show()

        # Correlation and nulls
        val_corr[k], pvals[k], rnulls[k], _ = compute_r_and_nulls(cancer_parc, nulls, Ws[k])
        print(f"{title_cancer} {names[k]}  r={val_corr[k]}  p={pvals[k]}")

    print('-----------------------')
    return val_corr, pvals, rnulls

# Run
breast_r, breast_p, breast_rnull = main_function(breast_data, breast_neighbour_abnormality, breast_nulls, 'Breast_node_neighbour')
lung_r, lung_p, lung_rnull       = main_function(lung_data, lung_neighbour_abnormality, lung_nulls, 'Lung_node_neighbour')
mel_r, mel_p, mel_rnull          = main_function(melanoma_data, melanoma_neighbour_abnormality, melanoma_nulls, 'Melanoma_node_neighbour')

#------------------------------------------------------------------------------
# scatter grid
#------------------------------------------------------------------------------

make_scatter_3x5 = True
if make_scatter_3x5:
    fig, axes = plt.subplots(3, 5, figsize=(24, 13))
    cancers = [("Breast", breast_data, breast_neighbour_abnormality, breast_r, breast_p),
               ("Lung", lung_data, lung_neighbour_abnormality, lung_r, lung_p),
               ("Melanoma", melanoma_data, melanoma_neighbour_abnormality, mel_r, mel_p)]
    cols = [(0, "DIST"), (1, "FC"), (2, "ATT"), (3, "CBF"), (4, "GENE")]
    for i, (cname, cvec, neigh, rr, pp) in enumerate(cancers):
        for j, (idx, wname) in enumerate(cols):
            ax = axes[i, j]
            sns.regplot(x=neigh[:, idx], y=cvec, ax=ax,
                        color='silver',
                        scatter_kws={'s': 25, 'edgecolor': 'black'},
                        fit_reg=False)
            ax.set_title(f"{cname} | {wname}\nr={rr[idx]:.3f}, p={pp[idx]:.4f}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_results + "ScatterGrid_3x5_DIST_FC_ATT_CBF_GENE.svg", format="svg")
    plt.show()

#------------------------------------------------------------------------------

make_boxplot_row_15 = True
if make_boxplot_row_15:
    # order: DIST(B,L,M), FC(B,L,M), ATT(B,L,M), CBF(B,L,M), GENE(B,L,M)
    rnull_sets = [breast_rnull[0], lung_rnull[0], mel_rnull[0],   # DIST
                  breast_rnull[1], lung_rnull[1], mel_rnull[1],   # FC
                  breast_rnull[2], lung_rnull[2], mel_rnull[2],   # ATT
                  breast_rnull[3], lung_rnull[3], mel_rnull[3],   # CBF
                  breast_rnull[4], lung_rnull[4], mel_rnull[4],   # GENE
                  ]

    rreal_sets = np.array([breast_r[0], lung_r[0], mel_r[0],
                           breast_r[1], lung_r[1], mel_r[1],
                           breast_r[2], lung_r[2], mel_r[2],
                           breast_r[3], lung_r[3], mel_r[3],
                           breast_r[4], lung_r[4], mel_r[4]])

    p_sets = np.array([breast_p[0], lung_p[0], mel_p[0],
                       breast_p[1], lung_p[1], mel_p[1],
                       breast_p[2], lung_p[2], mel_p[2],
                       breast_p[3], lung_p[3], mel_p[3],
                       breast_p[4], lung_p[4], mel_p[4]])

    labels = ["DIST\nBreast","DIST\nLung","DIST\nMel",
              "FC\nBreast","FC\nLung","FC\nMel",
              "ATT\nBreast","ATT\nLung","ATT\nMel",
              "CBF\nBreast","CBF\nLung","CBF\nMel",
              "GENE\nBreast","GENE\nLung","GENE\nMel"]

    all_vals = np.concatenate([np.concatenate(rnull_sets), rreal_sets])
    ymin, ymax = np.min(all_vals), np.max(all_vals)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin, ymax = ymin - pad, ymax + pad

    fig, axes = plt.subplots(1, 15, figsize=(36, 5), sharey=True)
    for i, ax in enumerate(axes):
        sns.boxplot(y=rnull_sets[i], ax=ax, color="lightgray", width=0.4)
        ax.scatter([0], [rreal_sets[i]], s=70, color="black", zorder=5)
        ax.set_ylim([ymin, ymax])
        ax.set_title(f"{labels[i]}\n"
                     f"r={rreal_sets[i]:.3f}, p={p_sets[i]:.4f}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.set_ylabel("Null r")
            ax.spines["left"].set_visible(True)
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.set_ylabel("")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(path_results + "NullBoxRow_15panels_groupedByFeature.svg", format="svg")
    plt.show()

#------------------------------------------------------------------------------
# FDR correction for the 15 tests
#------------------------------------------------------------------------------

p_all = np.array([breast_p[0], lung_p[0], mel_p[0],
                  breast_p[1], lung_p[1], mel_p[1],
                  breast_p[2], lung_p[2], mel_p[2],
                  breast_p[3], lung_p[3], mel_p[3],
                  breast_p[4], lung_p[4], mel_p[4]])

reject, q_all, _, _ = multipletests(p_all, alpha=0.05, method="fdr_by")

#------------------------------------------------------------------------------
# END