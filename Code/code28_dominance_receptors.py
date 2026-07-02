"""
###############################################################################

Dominance analysis

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from netneurotools import stats
from scipy.stats import zscore, pearsonr
from globals import path_results, path_file
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform, pdist
from nilearn.datasets import fetch_atlas_schaefer_2018

#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

nnodes = 400
nspins = 1000

#------------------------------------------------------------------------------
# Load cancer data
#------------------------------------------------------------------------------

cancer_type = 'Melanoma'
data_cancer = np.load(path_results + f'{cancer_type}_400_surf.npy').flatten()
null_data = np.load(path_results + f'cortical_nulls_{cancer_type}_1k_surf_brainspace.npy').T

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def get_reg_r_sq(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared

def cv_slr_distance_dependent(X, y, coords, train_pct = .75, metric = 'rsq'):
    
    '''
    cross validates linear regression model using distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1), percent of observations in training set
    metric = {'rsq', 'corr'}
    '''
    P = squareform(pdist(coords, metric = "euclidean"))
    train_metric = []
    test_metric = []

    for i in range(len(y)):
        distances = P[i, :]  # for every node
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])
        if metric == 'rsq':
            # get r^2 of train set
            train_metric.append(get_reg_r_sq(X[train_idx, :], y[train_idx]))

        elif metric == 'corr':
            rho, _ = pearsonr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])
        if metric == 'rsq':
            
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1-(1-r_squared)*((len(y[test_idx]) - 1) /
                                                  (len(y[test_idx]) -
                                                   X.shape[1] - 1))
            test_metric.append(adjusted_r_squared)

        elif metric == 'corr':
            rho, _ = pearsonr(yhat, y[test_idx])
            test_metric.append(rho)
    return train_metric, test_metric

def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.nanmean(null))
                    > abs(emp - np.nanmean(null)))) / (len(null) + 1)

def get_reg_r_pval(X, y, spins, nspins):
    
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(spins[:, s].T.reshape(-1, 1), y)
    return (1 + sum(null > emp))/(nspins + 1)

#------------------------------------------------------------------------------
# Load neurotransmitters
#------------------------------------------------------------------------------

parc_file_mni = fetch_atlas_schaefer_2018(n_rois = nnodes)['maps']
cortex = np.arange(nnodes)

receptor_data = pd.read_csv(path_file + 'receptor_data_scale400.csv', header = None)
data_names = np.load(path_file + 'receptor_names_pet.npy')
data = np.array(receptor_data)  # 400 by 18
num_receptors = len(data_names)

#------------------------------------------------------------------------------
#                              Dominance Analysis
#------------------------------------------------------------------------------

model_metrics = dict([])
train_metric = np.zeros([nnodes, 1])
test_metric = np.zeros(train_metric.shape)

nspins = 100
coords = np.genfromtxt(path_file + 'Schaefer_400.txt')
coords = coords[:, -3:]

model_metrics, _ = stats.get_dominance_stats(zscore(data),
                                             zscore(data_cancer))
# Cross validate the model
[train_metric , test_metric] = cv_slr_distance_dependent(zscore(data), 
                                                         zscore(data_cancer),
                                                         coords,
                                                         0.75,
                                                         metric = 'rsq')
# Get p-value of model
model_pval = get_reg_r_pval(zscore(data),
                            zscore(data_cancer),
                            null_data,
                            nspins)

dominance = np.zeros((1, len(data_names)))
dominance[0, :] = model_metrics["total_dominance"]

#------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------

plt.ion()
plt.figure()
plt.bar(np.arange(1), 
        np.sum(dominance, axis = 1),
        tick_label = 'perfusion score map')
plt.xticks(rotation = 'vertical')
plt.tight_layout()
plt.savefig(path_results + f'{cancer_type}rev1_dominance_molecular_2.svg', dmi = 300)
dominance[np.where(model_pval >= 0.05)[0], :] = 0

plt.ion()
plt.figure()
sns.heatmap(dominance / np.sum(dominance, axis = 1)[:, None],
            cmap = 'coolwarm',
            vmin= -0.15, vmax = 0.15,
            xticklabels = data_names,linewidth = 1.5)
plt.savefig(path_results + f'{cancer_type}rev1_dominance_molecular_1.svg', dmi = 300)
plt.tight_layout()

# plot cross validation results
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.violinplot(data = train_metric, ax = ax1)
sns.violinplot(data = test_metric, ax = ax2)
ax1.set(ylabel = 'train set correlation', ylim = (-1, 1))
ax2.set_xticklabels('perfusion score map', rotation = 90)
ax2.set(ylabel = 'test set correlation', ylim = (-1, 1))
plt.tight_layout()
plt.savefig(path_results + f'{cancer_type}_dominance_molecular.svg', dmi = 300)

#------------------------------------------------------------------------------
# END