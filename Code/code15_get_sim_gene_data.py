"""
###############################################################################

Calculate the gene similarity map for both cortex and cerebellum

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from globals import path_results

#------------------------------------------------------------------------------
# Load gene data
#------------------------------------------------------------------------------

gene_name = np.load(path_results + 'names_genes.npy', allow_pickle=True)
expression_val = np.load(path_results + 'gene_coexpression.npy', allow_pickle=True)

n_labels, n_genes = expression_val.shape

#------------------------------------------------------------------------------
# Calculate the similarity network
#------------------------------------------------------------------------------

sim = np.corrcoef(expression_val.T, rowvar=False)
np.fill_diagonal(sim, 1.0)

np.save(path_results + 'gene_similarity_432.npy', sim)

#------------------------------------------------------------------------------
# Show the similarity netwrok as a heatmap
#------------------------------------------------------------------------------

plt.figure(figsize=(7, 6))
im = plt.imshow(sim,vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson r")
plt.title("Gene-expression similarity across parcels")
plt.xlabel("Parcels")
plt.ylabel("Parcels")
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# END