"""
###############################################################################

Create 10k Spearman null correlation matrices

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
from scipy.stats import rankdata
from globals import path_results

#------------------------------------------------------------------------------
# Load gene data
#------------------------------------------------------------------------------

gene_name = np.load(path_results + 'names_genes.npy', allow_pickle=True)
expression_val = np.load(path_results + 'gene_coexpression.npy', allow_pickle=True)[:,:]
n_parcels, n_genes = expression_val.shape

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

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

def spearmanr_nulls_vs_genes_fast(nulls, exp, chunk_size=1000):
    """
    nulls: (N, K)
    exp:   (N, G)
    returns: (K, G)
    """
    nulls = np.asarray(nulls, dtype=float)
    exp = np.asarray(exp, dtype=float)
    n_nulls = nulls.shape[1]
    R = np.empty((n_nulls, exp.shape[1]), dtype=np.float32)
    exp_rank = rank_columns(exp)
    for start in range(0, n_nulls, chunk_size):
        end = min(start + chunk_size, n_nulls)
        print(f'processing nulls {start}:{end} / {n_nulls}', flush=True)
        null_chunk = nulls[:, start:end]
        null_chunk_rank = rank_columns(null_chunk)
        R[start:end, :] = pearsonr_matrix(null_chunk_rank, exp_rank).astype(np.float32)
    return R

#------------------------------------------------------------------------------
# Calculate null corr for each cancer type
#------------------------------------------------------------------------------

print('start for Breast cancer (10k spearman)', flush=True)
breast_nulls = np.load(path_results + 'nulls_Breast_10k_surf_brainspace.npy').T
R_breast = spearmanr_nulls_vs_genes_fast(breast_nulls, expression_val, chunk_size=1000)
np.save(path_results + 'Breast_null_corr_spearmanr_10k.npy', R_breast)

print('start for Lung cancer (10k spearman)', flush=True)
lung_nulls = np.load(path_results + 'nulls_Lung_10k_surf_brainspace.npy').T
R_lung = spearmanr_nulls_vs_genes_fast(lung_nulls, expression_val, chunk_size=1000)
np.save(path_results + 'Lung_null_corr_spearmanr_10k.npy', R_lung)

print('start for Melanoma cancer (10k spearman)', flush=True)
mel_nulls = np.load(path_results + 'nulls_Melanoma_10k_surf_brainspace.npy').T
R_mel = spearmanr_nulls_vs_genes_fast(mel_nulls, expression_val, chunk_size=1000)
np.save(path_results + 'Melanoma_null_corr_spearmanr_10k.npy', R_mel)

#------------------------------------------------------------------------------
# END