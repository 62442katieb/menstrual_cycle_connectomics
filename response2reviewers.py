import json

import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join
from idconn import io
from scipy.stats import spearmanr, pointbiserialr, pearsonr

def calc_fd(confounds):
    x = confounds["trans_x"].values
    y = confounds["trans_y"].values
    z = confounds["trans_z"].values
    alpha = confounds["rot_x"].values
    beta = confounds["rot_y"].values
    gamma = confounds["rot_z"].values

    delta_x = [np.abs(t - s) for s, t in zip(x, x[1:])]
    delta_y = [np.abs(t - s) for s, t in zip(y, y[1:])]
    delta_z = [np.abs(t - s) for s, t in zip(z, z[1:])]

    delta_alpha = [np.abs(t - s) for s, t in zip(alpha, alpha[1:])]
    delta_beta = [np.abs(t - s) for s, t in zip(beta, beta[1:])]
    delta_gamma = [np.abs(t - s) for s, t in zip(gamma, gamma[1:])]

    fd = np.sum([delta_x, delta_y, delta_z, delta_alpha, delta_beta, delta_gamma], axis=0)
    return fd

# Opening JSON file 
f = open('paths.json') 

# returns JSON object as a list 
data = json.load(f) 

diva = data['diva']
andme = data['andme']

layout = bids.BIDSLayout(diva, derivatives=True)
dat = io.read_corrmats(layout, task="rest", deriv_name="IDConn", atlas="craddock2012", z_score=False)

layout2 = bids.BIDSLayout(andme, derivatives=True)
dat2 = io.read_corrmats(layout2, task="rest", deriv_name="IDConn", atlas="craddock2012", z_score=False)

for i in dat.index:
    try:
        dat.at[i, 'mean_FC'] = dat.loc[i]['adj'].mean().mean()
        dat.at[i, 'dset'] = 1
    except:
        pass

for i in dat2.index:
    try:
        dat2.at[i, 'mean_FC'] = dat2.loc[i]['adj'].mean().mean()
        dat2.at[i, 'dset'] = 0
    except:
        pass

mini_df = pd.concat(
    [dat, dat2],
    axis=0
)[
    [
        'dset', 
        'bc', 
        'estradiol', 
        'progesterone', 
        'framewise_displacement', 
        'mean_FC'
    ]
]

mini_df['framewise_displacement'].to_csv(
    '../data/framewise_displacement.csv'
)

mini_df = mini_df.dropna()

corrs = pd.DataFrame(
    index=mini_df.columns,
    columns=mini_df.columns
)

pvals = pd.DataFrame(
    index=mini_df.columns,
    columns=mini_df.columns
)

for col1 in mini_df.columns:
    for col2 in mini_df.columns:
        if col1 != col2:
            if len(mini_df[col1].unique() < 3) or len(mini_df[col1].unique() < 3):
                r,p = pointbiserialr(mini_df[col1], mini_df[col2])
            else:
                r,p = pearsonr(mini_df[col1], mini_df[col2])
            corrs.at[col1, col2] = r
            pvals.at[col1, col2] = p

corrs.to_csv(
    '../data/variable_correlations.csv'
)

pvals.to_csv(
    '../data/variable_correlations-p.csv'
)

