
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import bct
import bids
from os import makedirs
from os.path import join, exists
from nilearn import plotting, connectome
from scipy.stats import pearsonr, ttest_ind, ttest_rel, pointbiserialr
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np
import networkx as nx
import pandas as pd
import bct 
from scipy.stats import skew


def scale_free_tau(corrmat, skew_thresh=0.1, proportional=True):
    ''''
    Calculates threshold at which network becomes scale-free, estimated from the skewness of the networks degree distribution.
    Parameters
    ----------
    corrmat : numpy.array
        Correlation or other connectivity matrix from which tau_connected will be estimated.
        Should be values between 0 and 1.
    proportional : bool
        Determines whether connectivity matrix is thresholded proportionally or absolutely.
        Default is proportional as maintaining network density across participants is a priority
    Returns
    -------
    tau : float
        Lowest vaue of tau (threshold) at which network is scale-free.
    '''
    tau = 0.99
    skewness = 0
    while skewness < skew_thresh:
        if proportional:
            w = bct.threshold_proportional(corrmat, tau)
        else:
            w = bct.threshold_absolute(corrmat, tau)
        skewness = skew(bct.degrees_und(w))
        #print(skewness)
        tau -= 0.01
    return tau

def connected_tau(corrmat, proportional=True):
    '''
    Calculates threshold at network becomes node connected, using NetworkX's `is_connected` function.
    Parameters
    ----------
    corrmat : numpy.array
        Correlation or other connectivity matrix from which tau_connected will be estimated.
        Should be values between 0 and 1.
    proportional : bool
        Determines whether connectivity matrix is thresholded proportionally or absolutely.
        Default is proportional as maintaining network density across participants is a priority
    Returns
    -------
    tau : float
        Lowest vaue of tau (threshold) at which network ceases to be node-connected.
    '''
    tau = 0.01
    connected = False
    while connected == False:
        if proportional:
            w = bct.threshold_proportional(corrmat, tau)
        else:
            w = bct.threshold_absolute(corrmat, tau)
        w_nx = nx.convert_matrix.from_numpy_array(w)
        connected = nx.algorithms.components.is_connected(w_nx)
        tau += 0.01
    return tau


sns.set(font_scale=2, style='ticks', context='paper')
#sns.set(style='whitegrid', context='talk')
#plt.rcParams["font.family"] = "monospace"
#plt.rcParams['font.monospace'] = 'Courier New'
crayons = sns.crayon_palette(['Cotton Candy', 'Carnation Pink', 'Salmon', 'Pink Sherbert'])
sns.palplot(crayons)


import json
# Opening JSON file 
f = open('paths.json') 

# returns JSON object as a list 
data = json.load(f) 

diva = data['diva']
andme = data['andme']

overleaf_figs = data['overleaf_figs']


diva_sub = ['Bubbles', 'Buttercup', 'Blossom']
andme_sub = ['01']
subjects = diva_sub + andme_sub
bc_sub = ['Blossom', 'Buttercup']
non_bc = ['01', 'Bubbles']


diva_df = pd.read_csv(join(diva, 'participants.csv'), index_col=[0,1], usecols=['subject', 'session', 'Mean E2', 'Mean P4', 'bc', 'menst_cycle-day'])
andme_df = pd.read_csv(join(andme, 'participants.tsv'), sep='\t', index_col=[0,1], usecols=['participant_id','ses_id','estradiol', 'progesterone'])


# standardize estradiol and progesterone measures
def normalize(data):
    norm = (data - np.mean(data)) / np.std(data)
    return norm

andme_df['e2_norm'] = normalize(andme_df['estradiol'])
andme_df['p4_norm'] = normalize(andme_df['progesterone'])
andme_df['menst_cycle-day'] = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                               12, 13, 14, 15, 16, 17, 18, 19, 20,
                               1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 
                               22, 23, 24, 25, 26, 27, 28, 29, 30]
diva_df['e2_norm'] = normalize(diva_df['Mean E2'])
diva_df['p4_norm'] = normalize(diva_df['Mean P4'])


ppt_df = pd.concat([diva_df.rename({'Mean E2': 'estradiol', 
                                     'Mean P4': 'progesterone',
                                     #'psqi_total': 'Pittsburgh_Sleep',
                                     #'pss_total': 'Perceived_Stress',
                                     }, axis=1), 
                    andme_df.rename({'sub-01': '01'}, axis=0)])


ppt_df['subject'] = 'nan'
ppt_df['session'] = 'nan'
for i in ppt_df.index:
    ppt_df.at[i, 'subject'] = i[0]
    ppt_df.at[i, 'session'] = i[1]
    if i[0] in diva_sub:
        ppt_df.at[i,'FIU'] = True
    elif i[0] in andme_sub:
        ppt_df.at[i,'FIU'] = False
        if int(i[1].split('-')[1]) <= 30:
            ppt_df.at[i,'bc'] = 0
        elif int(i[1].split('-')[1]) > 30:
            ppt_df.at[i,'bc'] = 1



# ## step one: correlations between linearized upper triangles


sessions = {}
for subject in diva_sub:
    sessions[subject] = list(ppt_df[ppt_df['subject'] == subject]['session'])


ses = []
nums = np.arange(1,61)
for i in nums:
    if len(str(i)) < 2:
        ses.append(f'ses-0{i}')
    else:
        ses.append(f'ses-{i}')
sessions['01'] = ses

sub_ses = []
for key in sessions.keys():
    for item in sessions[key]:
        sub_ses.append((key,item))


corrmat_dict = {}

for subject in subjects:
    for session in sessions[subject]:
        fname = f'sub-{subject}_{session}_task-rest_space-MNI152NLin2009cAsym_atlas-craddock2012_tcorr05_2level_270_2mm-3d_desc-corrmat_bold.tsv'
        if subject == '01':
            root = andme
        else:
            root = diva
        try:
            corrmat = pd.read_csv(join(root, 
                                       'derivatives', 
                                       'IDConn', 
                                       f'sub-{subject}', 
                                       session, 
                                       'func', 
                                       fname),
                                  sep='\t', index_col=0, header=0)
            assert corrmat.values.shape == (268,268), f"corrmat wrong shape: is {corrmat.values.shape}, should be (268,268)"
            corrmat_triu = np.triu(corrmat.values)
            corrmat_dict[(subject, session)] = np.ravel(corrmat_triu, order='C')
        except Exception as e:
            print(f"could not load corrmat because none exists for {subject}, {session}:", e)


corrmat_df = pd.DataFrame.from_dict(corrmat_dict).T
zeroes = corrmat_df.sum() == 0

cat_sim = corrmat_df[zeroes[zeroes == False].index].T.corr()

cat_sim_z = np.arctanh(cat_sim.replace(1,0))
cat_sim_z.replace(0, np.nan, inplace=True)



fig,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cat_sim_z, square=True, cmap='Spectral_r', cbar_kws={"shrink": .75})
plt.tight_layout(w_pad=0.6)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks([])
ax.set_yticks([])
#ax.set_yticklabels(['Bubbles', '', 'Buttercup', '', 'Blossom', 'sub-01', ''])

plt.show()

fig.savefig('ConnectivityCorrelations.svg')
fig.savefig('ConnectivityCorrelations.png', dpi=600, bbox_inches='tight')
#fig.savefig(join('..', '4-corrmatcorrmat.png'), 
#            dpi=600)


# a. due to group = same group; diff individuals, bc use, and site
group_corrs = []
for i1 in cat_sim:
    for i2 in cat_sim:
        if ppt_df.loc[i1]['subject'] != ppt_df.loc[i2]['subject']:
            if ppt_df.loc[i1]['FIU'] != ppt_df.loc[i2]['FIU']:
                if ppt_df.loc[i1]['bc'] != ppt_df.loc[i2]['bc']:
                    group_corrs.append(np.mean(np.mean(cat_sim_z.loc[i1][i2])))
group_corr = np.nanmean(np.asarray(group_corrs))
group_sdev = np.nanstd(np.array(group_corrs))

# b. due to site = same site; diff individuals & bc use
site_corrs = []
for i1 in cat_sim:
    for i2 in cat_sim:
        if ppt_df.loc[i1]['subject'] != ppt_df.loc[i2]['subject']:
            if ppt_df.loc[i1]['FIU'] == ppt_df.loc[i2]['FIU']:
                if ppt_df.loc[i1]['bc'] != ppt_df.loc[i2]['bc']:
                    site_corrs.append(np.mean(np.mean(cat_sim_z.loc[i1][i2])))
site_corr = np.nanmean(np.asarray(site_corrs))
site_sdev = np.nanstd(np.array(site_corrs))

# c. due to individual
subj_corrs = []
for i1 in cat_sim:
    for i2 in cat_sim:
        if ppt_df.loc[i1]['subject'] == ppt_df.loc[i2]['subject']:
            subj_corrs.append(np.mean(np.mean(cat_sim_z.loc[i1][i2])))
subj_corr = np.nanmean(np.asarray(subj_corrs))
subj_sdev = np.nanstd(np.array(subj_corrs))

# d. due to bc_use
bc_corrs = []
for i1 in cat_sim:
    for i2 in cat_sim:
        if ppt_df.loc[i1]['FIU'] != ppt_df.loc[i2]['FIU']:
            if ppt_df.loc[i1]['bc'] == ppt_df.loc[i2]['bc']:
                bc_corrs.append(np.mean(np.mean(cat_sim_z.loc[i1][i2])))
bc_corr = np.nanmean(np.array(bc_corrs))
bc_sdev = np.nanstd(np.array(bc_corrs))


print(f'''Variability due to group: {group_corr}\n\
Variability due to site: {site_corr}\n\
Variability due to individual: {subj_corr}\n\
Variability due to BC use: {bc_corr}''')


group = pd.Series(group_corrs, name='Group')
site = pd.Series(site_corrs, name='Site')
subj = pd.Series(subj_corrs, name='Individual')
bcu = pd.Series(bc_corrs, name='HC Use')


var_sources = pd.concat([group, site, bcu, subj], axis=1)
var_long = pd.melt(var_sources)


fig, ax = plt.subplots(figsize=(6,4), layout='constrained')
# Set the font name for axis tick labels to be Comic Sans
g = sns.boxenplot(data=var_long, x='variable', y='value', palette=crayons)
ax.set_ylim((0,1.5))
ax.set_ylabel('Network similarity: Z(r)')
ax.set_xlabel('Variable')
sns.despine(fig)
ax.set_xticklabels(['Group', 'Dataset', 'HC Use', 'Individual'])
fig.savefig('NetworkSimilarity.svg')
fig.savefig('NetworkSimilarity.png', dpi=600, bbox_inches='tight')
#fig.savefig(join(overleaf_figs, '4-network-similarity.png'), 
#            dpi=600, bbox_inches='tight')

print('BC, site:\t', ttest_ind(bcu.dropna(), site.dropna()))
print('BC, group:\t', ttest_ind(bcu.dropna(), group.dropna()))
print('BC, subject:\t', ttest_ind(bcu.dropna(), subj.dropna()))
print('Site, group:\t', ttest_ind(site.dropna(), group.dropna()))
print('Site, subject:\t', ttest_ind(site.dropna(), subj.dropna()))
print('Group, subject:\t', ttest_ind(group.dropna(), subj.dropna()))

# ## Assessing collinearity among predictor variables


nonbrain_corr = pd.DataFrame(index=list(ppt_df.columns), columns=list(ppt_df.columns))
nonbrain_pval = pd.DataFrame(index=list(ppt_df.columns), columns=list(ppt_df.columns))
nonbrain_star = pd.DataFrame(index=list(ppt_df.columns), columns=list(ppt_df.columns))


subj_dumb = pd.get_dummies(ppt_df['subject'], prefix='sub-')
ppt_df = pd.concat([ppt_df, subj_dumb], axis=1)


#ppt_df.drop(['Pittsburgh_Sleep', 'Perceived_Stress'], axis=1, inplace=True)


for var1 in ppt_df.columns:
    for var2 in ppt_df.columns:
        if var1 != var2:
            #print(var1, var2)
            try:
                df = ppt_df[[var1,var2]].dropna()
                if np.unique(df[var1]).shape[0] == 2:
                    r, p = pointbiserialr(df[var1], df[var2])
                elif np.unique(df[var2]).shape[0] == 2:
                    r, p = pointbiserialr(df[var1], df[var2])
                else:
                    r, p = pearsonr(df[var1], df[var2])
                if p < 0.01:
                    nonbrain_star.at[var1, var2] = '0'
                    if p < 0.001:
                        print(f'{var1}, {var2}, r = {np.round(r, 2)}, p = {np.round(p, 4)}')
                        nonbrain_star.at[var1, var2] = '1'
                nonbrain_corr.at[var1, var2] = r
                nonbrain_pval.at[var1, var2] = p
            except Exception as e:
                print(var1, var2, e)


nonbrain_corr.dropna(how='all', axis=0, inplace=True)
nonbrain_corr.dropna(how='all', axis=1, inplace=True)
#nonbrain_corr.drop('BC_No', axis=1, inplace=True)
#nonbrain_corr.drop('BC_No', axis=0, inplace=True)
nonbrain_pval.dropna(how='all', axis=0, inplace=True)
nonbrain_pval.dropna(how='all', axis=1, inplace=True)


nonbrain_corr.rename({'estradiol': 'Estradiol', 
                      'progesterone': 'Progesterone', 
                      'menst_cycle-day': 'Menstrual Cycle Day', 
                      'FIU': 'Site', 
                      'BC_Yes': 'HC Use'}, axis=1, inplace=True)
nonbrain_corr.rename({'estradiol': 'Estradiol', 
                      'progesterone': 'Progesterone', 
                      'menst_cycle-day': 'Menstrual Cycle Day', 
                      'FIU': 'Site', 
                      'BC_Yes': 'HC Use'}, axis=0, inplace=True)


nonbrain_corr.to_csv('nonbrain-correlations.csv')
nonbrain_pval.to_csv('nonbrain-correlations-p.csv')


nonbrain_pval


nonbrain_star.dropna(how='all', axis=0, inplace=True)
nonbrain_star.dropna(how='all', axis=1, inplace=True)
nonbrain_star.fillna('ns', inplace=True)


mask = np.triu(np.ones_like(nonbrain_corr.values, dtype=bool))

fig,ax = plt.subplots(figsize=(15,15))
plt.tight_layout(w_pad=10)
sns.heatmap(nonbrain_corr.fillna(0), 
            cmap='RdBu_r', center=0, 
            mask=mask,
            square=True, linewidths=.5, 
            cbar_kws={"shrink": .83})
plt.tight_layout(w_pad=10)
fig.savefig('NonbrainCorrmat.svg')
fig.savefig('NonbrainCorrmat.png', dpi=300)

