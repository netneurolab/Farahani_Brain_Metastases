"""
###############################################################################

Gene expression analysis

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from globals import path_results, path_csv
from statsmodels.stats.multitest import multipletests

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def parse_genes(s):
    s = '' if pd.isna(s) else str(s).strip()
    if not s:
        return []
    parts = re.split(r'\s*[|,;]\s*', s)
    return [g.strip().upper() for g in parts if g.strip()]

def pearsonr_matrix(X, Y):
    """
    X: (N, K)
    Y: (N, G)
    returns: (K, G)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    Xstd = Xc.std(axis=0, ddof=1, keepdims=True)
    Ystd = Yc.std(axis=0, ddof=1, keepdims=True)

    Xstd[Xstd == 0] = np.nan
    Ystd[Ystd == 0] = np.nan

    Xz = Xc / Xstd
    Yz = Yc / Ystd

    return (Xz.T @ Yz) / (X.shape[0] - 1)

def rank_columns(A):
    """
    Rank each column independently.
    """
    return np.apply_along_axis(rankdata, 0, np.asarray(A, dtype=float))

def two_sided_perm_p(obs, null):
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    if len(null) == 0 or not np.isfinite(obs):
        return np.nan, np.nan

    p = (np.sum(((null - np.mean(null))) >= ((obs- np.mean(null)))) + 1) / (len(null) + 1)
    return p, np.mean(null- np.mean(null))

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

gene_name = np.load(path_results + 'names_genes.npy', allow_pickle=True)
expression_val = np.load(path_results + 'gene_coexpression.npy', allow_pickle=True)
n_labels, n_genes = expression_val.shape

cancer_name = 'Melanoma'
cancer_parc_cereb = np.load(path_results + cancer_name + '_34_cerebellum.npy')
cancer_parc_cortex = np.load(path_results + cancer_name + '_400_surf.npy').flatten()
cancer_parc = np.concatenate((cancer_parc_cereb[:32], cancer_parc_cortex), axis=0)
null_corrs_z = np.load(path_results + cancer_name + '_null_corr_spearmanr_10k.npy')

#------------------------------------------------------------------------------
# Gene-wise phenotype correlations
#------------------------------------------------------------------------------

ranked_cancer_parc = rankdata(cancer_parc).reshape(-1, 1)
ranked_expression = rank_columns(expression_val)
corr_z = pearsonr_matrix(ranked_cancer_parc, ranked_expression).flatten().astype(np.float32)

df_corr = pd.DataFrame({"gene": gene_name, "z_corr": corr_z}).reset_index(drop=True)

#------------------------------------------------------------------------------
# Load pathway / gene-set table
#------------------------------------------------------------------------------

df = pd.read_csv(path_csv +  'GCEA_GO-biologicalProcessProp-discrete.csv')
df['genes_names'] = df['cGenes'].apply(parse_genes)

# Background genes = genes with finite observed z-correlation
genes_upper = np.array([str(g).upper() for g in np.ravel(gene_name)])

# keep genes with finite observed z and finite nulls across all permutations
finite_obs = np.isfinite(corr_z)
finite_null = np.all(np.isfinite(null_corrs_z), axis=0)
keep = finite_obs & finite_null
genes_keep = genes_upper[keep]
corr_z_keep = corr_z[keep]
null_corrs_z_keep = null_corrs_z[:, keep]
_, first_idx = np.unique(genes_keep, return_index=True)
first_idx = np.sort(first_idx)
genes_keep = genes_keep[first_idx]
corr_z_keep = corr_z_keep[first_idx]
null_corrs_z_keep = null_corrs_z_keep[:, first_idx]
gene_to_idx = {g: i for i, g in enumerate(genes_keep)}
gene_to_z = {g: z for g, z in zip(genes_keep, corr_z_keep)}

#------------------------------------------------------------------------------
# Category enrichment
#------------------------------------------------------------------------------

n_nulls = null_corrs_z_keep.shape[0]
n_pathways = len(df)

obs_score = np.full(n_pathways, np.nan)
null_mean_score = np.full(n_pathways, np.nan)
pvals = np.full(n_pathways, np.nan)
tail_used = np.full(n_pathways, None, dtype=object)

k_input = np.zeros(n_pathways, dtype=int)
k_present = np.zeros(n_pathways, dtype=int)
k_missing = np.zeros(n_pathways, dtype=int)

for i, row in df.iterrows():
    gs_all = np.unique(np.asarray(row['genes_names'], dtype=object))
    gs_all = gs_all[gs_all != '']

    k_input[i] = len(gs_all)

    gs_present = np.array([g for g in gs_all if g in gene_to_idx], dtype=object)
    k_present[i] = len(gs_present)
    k_missing[i] = k_input[i] - k_present[i]

    if k_present[i] == 0:
        continue

    idxs = np.array([gene_to_idx[g] for g in gs_present], dtype=int)

    # observed category score
    obs_score[i] = np.mean(corr_z_keep[idxs])

    # null category scores
    null_scores = np.mean(null_corrs_z_keep[:, idxs], axis=1)

    pvals[i], null_mean_score[i] = two_sided_perm_p(obs=obs_score[i], null=null_scores)
    tail_used[i] = 'two-sided'

#------------------------------------------------------------------------------
# Organize results
#------------------------------------------------------------------------------

res = df[['cDesc', 'cSize', 'cGenes']].copy()
res['k_input_from_cGenes'] = k_input
res['k_present'] = k_present
res['k_missing'] = k_missing
res['fraction_present'] = np.divide(
    k_present,
    k_input,
    out=np.full(n_pathways, np.nan, dtype=float),
    where=k_input > 0)
res['category_score_z'] = obs_score
res['null_mean_score_z'] = null_mean_score
res['p_one_sided'] = pvals
res['tail'] = tail_used

# Minimum pathway size
res = res[res['k_present'] >= 5].copy()

# FDR only on finite p-values
res['p_corrected'] = np.nan
valid = np.isfinite(res['p_one_sided'].values)
res.loc[valid, 'p_corrected'] = multipletests(
    res.loc[valid, 'p_one_sided'].values,
    method='fdr_tsbh')[1]

# Sort by strongest positive score first
res = res.sort_values(
    by=['category_score_z', 'p_one_sided'],
    ascending=[False, True],
    na_position='last').reset_index(drop=True)

res_sig = res[res['p_corrected'] <= 0.05].copy()
res.to_csv(path_results + f'{cancer_name}_GSEA_cells_results.csv', index=False)

#------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------

res_plot = (res.sort_values(by='p_one_sided', ascending=True, na_position='last')
            .head(15).sort_values(by='category_score_z', ascending=True).copy())
res_plot['significant'] = res_plot['p_corrected'] <= 0.05

# In RED if significant
colors = res_plot['significant'].map({True: 'lightcoral', False: 'silver'}).values
fig, ax = plt.subplots(figsize=(12, 10))
for i, x in enumerate(res_plot['category_score_z']):
    ax.hlines(y=i, xmin=0, xmax=x, color=colors[i],
              alpha=0.7, linestyles='dashed', linewidth=2)

ax.scatter(res_plot['category_score_z'], range(len(res_plot)),
           color=colors, s=50, zorder=3)

ax.axvline(0, color='k', linewidth=1)
ax.set_xlabel('Category score (z)')
ax.set_yticks(range(len(res_plot)))
ax.set_yticklabels(res_plot['cDesc'])
ax.set_title(f'Top 15 enriched terms: {cancer_name}')
sns.despine(fig, left=True, bottom=True)
plt.tight_layout()
plt.savefig(path_results + f'{cancer_name}_GSEA_top15_lollipop.svg',
            bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------
# END