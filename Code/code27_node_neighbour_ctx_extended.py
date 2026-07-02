"""
###############################################################################
Breast_node_neighbour recep  r=0.44534322853121044  p=0.3196803196803197
Breast_node_neighbour met  r=0.7300362435377936  p=0.004995004995004995
Breast_node_neighbour SC  r=0.395515185835946  p=0.46553446553446554
Breast_node_neighbour dist  r=-0.5215008820167764  p=0.5244755244755245
Breast_node_neighbour gene  r=0.4192288339935699  p=0.22977022977022976
Breast_node_neighbour per  r=0.4411122560870073  p=0.022977022977022976
Breast_node_neighbour att  r=0.47059278911165214  p=0.008991008991008992
Breast_node_neighbour FC  r=0.37336747562966477  p=0.45254745254745254
-----------------------
Lung_node_neighbour recep  r=0.6225288456091407  p=0.016983016983016984
Lung_node_neighbour met  r=0.7194025981147605  p=0.3086913086913087
Lung_node_neighbour SC  r=0.5168342160884796  p=0.06593406593406594
Lung_node_neighbour dist  r=-0.5538754066387425  p=0.9210789210789211
Lung_node_neighbour gene  r=0.5203812087404442  p=0.052947052947052944
Lung_node_neighbour per  r=0.5722188857320794  p=0.04595404595404595
Lung_node_neighbour att  r=0.5362979552553633  p=0.025974025974025976
Lung_node_neighbour FC  r=0.4256628222472442  p=0.6063936063936064
-----------------------
Melanoma_node_neighbour recep  r=0.4425358247464659  p=0.04895104895104895
Melanoma_node_neighbour met  r=0.6262328232154024  p=0.11888111888111888
Melanoma_node_neighbour SC  r=0.24976165334675438  p=0.5214785214785215
Melanoma_node_neighbour dist  r=-0.3677278346056484  p=0.7712287712287712
Melanoma_node_neighbour gene  r=0.2994222653807852  p=0.24375624375624375
Melanoma_node_neighbour per  r=0.18728192660217882  p=0.18081918081918083
Melanoma_node_neighbour att  r=0.16124002965844464  p=0.7842157842157842
Melanoma_node_neighbour FC  r=0.3110698566684224  p=0.3176823176823177
-----------------------

BH-FDR across all 9 tests:
recep
Breast    p=0.319680  q=0.479520
recep
Lung      p=0.016983  q=0.124675
recep
Mel       p=0.048951  q=0.158841


met
Breast      p=0.004995  q=0.107892
met
Lung        p=0.308691  q=0.479520
met
Mel         p=0.118881  q=0.285315


sc
Breast       p=0.465534  q=0.620713
sc
Lung         p=0.065934  q=0.175824
sc
Mel          p=0.521479  q=0.629371


dist
Breast     p=0.524476  q=0.629371
dist
Lung       p=0.921079  q=0.921079
dist
Mel        p=0.771229  q=0.818312


gene
Breast     p=0.229770  q=0.450012
gene
Lung       p=0.052947  q=0.158841
gene
Mel        p=0.243756  q=0.450012


per
Breast      p=0.022977  q=0.124675
per
Lung        p=0.045954  q=0.158841
per
Mel         p=0.180819  q=0.394515


att
Breast      p=0.008991  q=0.107892
att
Lung        p=0.025974  q=0.124675
att
Mel         p=0.784216  q=0.818312


fc
Breast       p=0.452547  q=0.620713
fc
Lung         p=0.606394  q=0.693021
fc
Mel          p=0.317682  q=0.479520

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
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from globals import path_results, path_FC
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

n_nodes = 400
nspins = 1000

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
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
    r_real, _ = pearsonr(map_vec, neigh_real)
    r_spins = np.zeros(nspins)
    for n in range(nspins):
        null_map = nulls[:, n]
        null_neigh = node_to_neighbour(null_map, W)
        r_spins[n], _ = pearsonr(null_map, null_neigh)
    p_spin = pval_cal(r_real, r_spins.reshape(-1, 1), nspins)
    return r_real, p_spin, r_spins, neigh_real

#------------------------------------------------------------------------------
# Load similarity matrices (ATT, CBF, Gene)
#------------------------------------------------------------------------------

recep = np.load('/home/afarahani/Desktop/met_data/Data/cortical_networks/receptor_similarity.npy')
recep[recep<0] = 0
np.fill_diagonal(recep,0)

met = np.load('/home/afarahani/Desktop/met_data/Data/cortical_networks/metabolic_connectivity.npy')
met[met<0] = 0
np.fill_diagonal(met,0)

sc = np.load('/home/afarahani/Desktop/met_data/Data/cortical_networks/laminar_similarity.npy')
sc[sc<0] = 0
np.fill_diagonal(sc,0)

atlas_path = os.path.join(path_results, 'cerebellum_cortex_atlas_09.nii.gz')
atlas = nib.load(atlas_path).get_fdata()
atlas = atlas - 2
atlas[atlas<=0] = 0

com = np.zeros((432, 3))
c = 0
for lab in np.unique(atlas)[1:]:
    temp = (atlas == lab)
    com[c, :] = ndi.center_of_mass(temp)
    c += 1
distance = cdist(com, com)  # (432,432)
distW = distance.copy()
distW = prep_W(distW, zero_diag=True, drop_neg=False)  # distance isn't negative anyway
distW = distW[32:, 32:]

gene_unmasked = np.load(path_results + 'gene_similarity_432.npy')[32:, 32:]
gene_unmasked[gene_unmasked<0] = 0
gene = gene_unmasked
gene = prep_W(gene, zero_diag=True, drop_neg=True)

per = np.load(path_results + 'zscored_blood_similarity_432.npy')[32:, 32:]
per = prep_W(per, zero_diag=True, drop_neg=True)

att = np.load(path_results + 'zscored_blood_arrival_similarity_432.npy')[32:, 32:]
att = prep_W(att, zero_diag=True, drop_neg=True)


fc_use_abs = False     # False -> "+ only" (drop negatives), True -> abs(FC)
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
FC = FC[32:, 32:]

#------------------------------------------------------------------------------
# Similarity matrix visualization: 5 heatmaps side-by-side
#------------------------------------------------------------------------------

fig, axes = plt.subplots(1, 8, figsize=(36, 6))
sns.heatmap(distW, vmin=-np.max(distW), vmax=np.max(distW),cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[0])
axes[0].set_title('distance')

sns.heatmap(FC, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[1])
axes[1].set_title('fc')

sns.heatmap(per, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[2])
axes[2].set_title('per')

sns.heatmap(att, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[3])
axes[3].set_title('att')

sns.heatmap(gene, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[4])
axes[4].set_title('gene')
sns.heatmap(met, vmin=-0.3, vmax=0.3, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[5])
axes[5].set_title('met')

sns.heatmap(recep, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[6])
axes[6].set_title('recep')

sns.heatmap(sc, vmin=-1, vmax=1, cmap='coolwarm',
            xticklabels=False, yticklabels=False, cbar=False, ax=axes[7])
axes[7].set_title('laminar')
plt.tight_layout()
plt.savefig(path_results + "Heatmaps_cortical.png", dpi=300)
plt.show()

#------------------------------------------------------------------------------
# Load cancer data
#------------------------------------------------------------------------------

breast_data = np.load(path_results + 'Breast_400_surf.npy').flatten()
breast_nulls = np.load(path_results + 'cortical_nulls_Breast_1k_surf_brainspace.npy').T

lung_data = np.load(path_results + 'Lung_400_surf.npy').flatten()
lung_nulls = np.load(path_results + 'cortical_nulls_Lung_1k_surf_brainspace.npy').T

melanoma_data = np.load(path_results + 'Melanoma_400_surf.npy').flatten()
melanoma_nulls = np.load(path_results + 'cortical_nulls_Melanoma_1k_surf_brainspace.npy').T

#------------------------------------------------------------------------------
# Node - neighbour
#------------------------------------------------------------------------------

def build_neighbour_matrix(map_vec):
    neigh = np.zeros((n_nodes, 8))
    for i in range(n_nodes):
        neigh[i, 0] = np.nansum(map_vec * recep[i, :])   / np.nansum(recep[i, :])     # recep
        neigh[i, 1] = np.nansum(map_vec * met[i, :])   / np.nansum(met[i, :])         # met
        neigh[i, 2] = np.nansum(map_vec * sc[i, :])   / np.nansum(sc[i, :])           # sc
        neigh[i, 3] = np.nansum(map_vec * distW[i, :])   / np.nansum(distW[i, :])     # distance
        neigh[i, 4] = np.nansum(map_vec * gene[i, :])   / np.nansum(gene[i, :])       # gene
        neigh[i, 5] = np.nansum(map_vec * per[i, :])   / np.nansum(per[i, :])         # per
        neigh[i, 6] = np.nansum(map_vec * att[i, :])   / np.nansum(att[i, :])         # att
        neigh[i, 7] = np.nansum(map_vec * FC[i, :])   / np.nansum(FC[i, :])           # FC
    return neigh

breast_neighbour_abnormality = build_neighbour_matrix(breast_data)
lung_neighbour_abnormality   = build_neighbour_matrix(lung_data)
melanoma_neighbour_abnormality = build_neighbour_matrix(melanoma_data)

#------------------------------------------------------------------------------
# Main analysis function: compute r + spin p-values
#------------------------------------------------------------------------------
def main_function(cancer_parc, neighbour_abnormality, nulls, title_cancer):
    # Real r
    val_corr = np.zeros(8)
    pvals  = [None]*8
    rnulls = [None]*8

    names = [ "recep", "met", "SC", "dist", 'gene', 'per', 'att', 'FC']
    Ws    = [   recep,   met, sc, distW, gene, per, att, FC]

    for k in range(8):
        # Scatter as individual SVGs
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
# Scatter plots and regression lines
#------------------------------------------------------------------------------

make_scatter_3x5 = True

if make_scatter_3x5:
    fig, axes = plt.subplots(3,8, figsize=(30, 15))

    cancers = [("Breast", breast_data, breast_neighbour_abnormality, breast_r, breast_p),
               ("Lung", lung_data, lung_neighbour_abnormality, lung_r, lung_p),
               ("Melanoma", melanoma_data, melanoma_neighbour_abnormality, mel_r, mel_p)]

    cols = [(0, "recep"), (1, "met"), (2, "sc"),
            (3, "dist"), (4, "gene"), (5, "per"),
            (6, "att"), (7, "FC")]

    for i, (cname, cvec, neigh, rr, pp) in enumerate(cancers):
        for j, (idx, wname) in enumerate(cols):
            ax = axes[i, j]
            sns.regplot(x=neigh[:, idx], y=cvec, ax=ax,
                        color='silver',
                        scatter_kws={'s': 25, 'edgecolor': 'black'},
                        fit_reg=False)  # keep your "no line" choice
            ax.set_title(f"{cname} | {wname}\nr={rr[idx]:.3f}, p={pp[idx]:.4f}", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_results + "ScatterGrid_cortex.svg", format="svg")
    plt.show()

#------------------------------------------------------------------------------

make_boxplot_row_15 = True

if make_boxplot_row_15:

    # order: DIST(B,L,M), FC(B,L,M), ATT(B,L,M), CBF(B,L,M), GENE(B,L,M)
    rnull_sets = [breast_rnull[0], lung_rnull[0], mel_rnull[0],   # recep
                  breast_rnull[1], lung_rnull[1], mel_rnull[1],   # met
                  breast_rnull[2], lung_rnull[2], mel_rnull[2],   # sc
                  breast_rnull[3], lung_rnull[3], mel_rnull[3],   # dist
                  breast_rnull[4], lung_rnull[4], mel_rnull[4],   # gene
                  breast_rnull[5], lung_rnull[5], mel_rnull[5],   # per
                  breast_rnull[6], lung_rnull[6], mel_rnull[6],   # att
                  breast_rnull[7], lung_rnull[7], mel_rnull[7]]   # FC

    rreal_sets = np.array([breast_r[0], lung_r[0], mel_r[0],
                           breast_r[1], lung_r[1], mel_r[1],
                           breast_r[2], lung_r[2], mel_r[2],
                           breast_r[3], lung_r[3], mel_r[3],
                           breast_r[4], lung_r[4], mel_r[4],
                           breast_r[5], lung_r[5], mel_r[5],
                           breast_r[6], lung_r[6], mel_r[6],
                           breast_r[7], lung_r[7], mel_r[7]])

    p_sets = np.array([breast_p[0], lung_p[0], mel_p[0],
                       breast_p[1], lung_p[1], mel_p[1],
                       breast_p[2], lung_p[2], mel_p[2],
                       breast_p[3], lung_p[3], mel_p[3],
                       breast_p[4], lung_p[4], mel_p[4],
                       breast_p[5], lung_p[5], mel_p[5],
                       breast_p[6], lung_p[6], mel_p[6],
                       breast_p[7], lung_p[7], mel_p[7]])

    labels = ["recep\nBreast","recep\nLung","recep\nMel",
              "met\nBreast","met\nLung","met\nMel",
              "sc\nBreast","sc\nLung","sc\nMel",
              "dist\nBreast","dist\nLung","dist\nMel",
              "gene\nBreast","gene\nLung","gene\nMel",
              "per\nBreast","per\nLung","per\nMel",
              "att\nBreast","att\nLung","att\nMel",
              "fc\nBreast","fc\nLung","fc\nMel"]

    all_vals = np.concatenate([np.concatenate(rnull_sets), rreal_sets])
    ymin, ymax = np.min(all_vals), np.max(all_vals)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ymin, ymax = ymin - pad, ymax + pad
    fig, axes = plt.subplots(1, 24, figsize=(16, 5), sharey=True)
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
    plt.savefig(path_results + "NullBoxRow_cortex.svg", format="svg")
    plt.show()

#------------------------------------------------------------------------------
# FDR correction (Benjamini–Hochberg) for the 9 tests
#------------------------------------------------------------------------------

p_all = np.array([
    breast_p[0], lung_p[0], mel_p[0],   # recep
    breast_p[1], lung_p[1], mel_p[1],   # met
    breast_p[2], lung_p[2], mel_p[2],   # sc / laminar
    breast_p[3], lung_p[3], mel_p[3],   # dist
    breast_p[4], lung_p[4], mel_p[4],   # gene
    breast_p[5], lung_p[5], mel_p[5],   # per / CBF
    breast_p[6], lung_p[6], mel_p[6],   # att
    breast_p[7], lung_p[7], mel_p[7],   # FC
], dtype=float)

reject, q_all, _, _ = multipletests(p_all, alpha=0.05, method="fdr_bh")

labels = ["recep\nBreast","recep\nLung","recep\nMel",
          "met\nBreast","met\nLung","met\nMel",
          "sc\nBreast","sc\nLung","sc\nMel",
          "dist\nBreast","dist\nLung","dist\nMel",
          "gene\nBreast","gene\nLung","gene\nMel",
          "per\nBreast","per\nLung","per\nMel",
          "att\nBreast","att\nLung","att\nMel",
          "fc\nBreast","fc\nLung","fc\nMel"]

print("\nBH-FDR across all 9 tests:")
for lab, p, q, rj in zip(labels, p_all, q_all, reject):
    star = "***" if rj else ""
    print(f"{star}{lab:14s}  p={p:.6f}  q={q:.6f}")

#------------------------------------------------------------------------------
# END