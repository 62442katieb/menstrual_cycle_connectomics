
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import bct
import bids
import semopy
from os import makedirs
from os.path import join, exists, basename
from nilearn import plotting, connectome
from scipy.stats import pearsonr, ttest_ind, ttest_rel, mannwhitneyu, pointbiserialr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import rc


def jili_sidak_mc(data, alpha):
    import math
    import numpy as np
    
    mc_corrmat = data.corr()
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('Number of effective comparisons: {0}'.format(M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff


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


sns.set(font_scale=1.2, style='ticks', context='paper')
crayons = sns.crayon_palette(['Cotton Candy', 'Carnation Pink', 'Salmon', 'Pink Sherbert'])

ppt_pal = sns.color_palette(['#685690', '#EA6964', '#33ACE3', '#4AB62C', ])

import json
# Opening JSON file 
f = open('paths.json') 

# returns JSON object as a list 
data = json.load(f) 

diva = data['diva']
andme = data['andme']

diva_sub = ['sub-02', 'sub-03', 'sub-04']
andme_sub = ['sub-01']
subjects = diva_sub + andme_sub
#bc_sub = ['Blossom', 'Buttercup']
#non_bc = ['01', 'Bubbles']

diva_df = pd.read_csv(join(diva, 'participants.csv'), index_col=[0,1])
andme_df = pd.read_table(join(andme, 'participants.tsv'), index_col=[0,1])

diva_df.rename(
    {
        'Blossom': 'sub-02',
        'Bubbles': 'sub-03',
        'Buttercup': 'sub-04',
    }, 
    inplace=True
)

diva_mapping = {
    'Blossom': 'sub-02',
    'Bubbles': 'sub-03',
    'Buttercup': 'sub-04',
    '01': 'sub-01'
}

fd_df = pd.read_csv('/Users/katherine.b/Dropbox/Projects/IDConn/diva_fd.csv', index_col=0)

# standardize estradiol and progesterone measures
#def normalize(data):
#    norm = (data - np.mean(data)) / np.std(data)
#    return norm

#andme_df['E2'] = normalize(andme_df['estradiol'])
#andme_df['P4'] = normalize(andme_df['progesterone'])

#diva_df['E2'] = normalize(diva_df['Mean E2'])
#diva_df['P4'] = normalize(diva_df['Mean P4'])

andme_df['E2'] = MinMaxScaler().fit_transform(andme_df['estradiol'].values.reshape(-1, 1))
andme_df['P4'] = MinMaxScaler().fit_transform(andme_df['progesterone'].values.reshape(-1, 1))

diva_df['E2'] = MinMaxScaler().fit_transform(diva_df['Mean E2'].values.reshape(-1, 1))
diva_df['P4'] = MinMaxScaler().fit_transform(diva_df['Mean P4'].values.reshape(-1, 1))

ppt_df = pd.concat([diva_df.rename({'Mean E2': 'estradiol', 
                                     'Mean P4': 'progesterone',
                                     'psqi_total': 'PittsburghSleep',
                                     'pss_total': 'PerceivedStress',
                                   'menst_cycle-day': 'cycle_day'}, axis=1), 
                    andme_df])


for i in ppt_df.index:
    ppt_df.at[i, 'subject'] = i[0]
    ppt_df.at[i, 'session'] = i[1]
ppt_df['E2XP4'] = ppt_df['E2'] * ppt_df['P4']

for i in fd_df.index:
    name_list = basename(i).split('_')
    subject = diva_mapping[name_list[0].split('-')[1]]
    session = name_list[1]
    ppt_df.at[(subject,session), 'fd'] = fd_df.loc[i].values[0]

small = ppt_df[['E2', 'P4', 'fd']].dropna()
print(pearsonr(small['E2'], small['fd']))
print(pearsonr(small['P4'], small['fd']))


ppt_df.drop(['RecordedDate_DateTime', 'asleep', 'pst_effort', 'sorpf_effort',
       'math_effort', 'eirt_effort', 'when_asleep', 'dusty_valence',
       'dusty_arousal', 'nancy_valence', 'nancy_arousal', 'steve_valence',
       'steve_arousal', 'jim_valence', 'jim_arousal', 'joyce_valence',
       'joyce_arousal', 'jonathan_valence', 'jonathan_arousal',
       'eleven_valence', 'eleven_arousal', 'lucas_valence', 'lucas_arousal',
       'mike_valence', 'mike_arousal', 'estradiol', 'progesterone', 'Mean Cort',
       'bc_name', 'bc_method', 'bc_elapse', 'menst_days',
       'menst_start-month', 'menst_start-day', 'menst_day-one',
       'PittsburghSleep', 'PerceivedStress',
       'panas_negative-affect', 'panas_positive-affect', 'panas_fear',
       'panas_hostility', 'panas_guilt', 'panas_sadness', 'panas_joviality',
       'panas_self-assurance', 'panas_attentiveness', 'panas_shyness',
       'panas_fatigue', 'panas_serenity', 'panas_surprise', 'lte_total',
       'saliva_time', 'ps_caffeine', 'ps_caffeine-elapse',
       'ps_caffeine-24h-mg', 'ps_caffeine-2h-mg', 'ps_cigarette-24h',
       'ps_aspirin-24h', 'ps_ibuprophen-24h', 'ps_ibuprophen-elapse_hr',
       'ps_ibuprophen-regular', 'ps_acetaminophen-24h', 
       'POMS_Tension', 'POMS_Depr', 'POMS_Anger', 'POMS_Vigor', 'POMS_Fatigue',
       'POMS_Confusion', 'POMS_total', 'Pittsburgh_Sleep', 'State_Anxiety',
       'Perceived_Stress', 'Total_Calories', 'dhea_s',
       'follicle_stimulating_hormone', 'luteinizing_hormone', 'shbg',
       'testosterone', 'menstrual_stage'], axis=1, inplace=True)


#ppt_df = pd.concat([ppt_df, 
#                    pd.get_dummies(ppt_df['bc_use'], 'HC')], 
#                   axis=1).drop(['bc_use', 'HC_No'],
#                                axis=1).rename({'HC_Yes': 'HC'}, axis=1)
ppt_df['HCUse'] = ppt_df['bc']
ppt_df['HCUse'] = ppt_df['HCUse'].replace({'0': 'Yes', '1': 'No'})


sessions = {}
for subject in diva_sub:
    sessions[subject] = list(ppt_df[ppt_df['subject'] == subject]['session'])


ses = []
nums = np.arange(1,31)
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


drop = ppt_df.filter(regex='ses.*b',axis=0).index

temp_df = ppt_df.drop(drop, axis=0).copy()

temp_df.at[('sub-04', 'ses-01'), 'cycle_day'] = 25
temp_df.at[('sub-04', 'ses-01b'), 'cycle_day'] = 28
temp_df.at[('sub-04', 'ses-02'), 'cycle_day'] = 3
temp_df.at[('sub-04', 'ses-02b'), 'cycle_day'] = 6
temp_df.at[('sub-04', 'ses-03'), 'cycle_day'] = 10
temp_df.at[('sub-04', 'ses-03b'), 'cycle_day'] = 13

temp_df.at[('sub-04', 'ses-04'), 'cycle_day'] = 17
temp_df.at[('sub-04', 'ses-04b'), 'cycle_day'] = 20
temp_df.at[('sub-04', 'ses-05a'), 'cycle_day'] = 24
temp_df.at[('sub-04', 'ses-05'), 'cycle_day'] = 27

temp_df['HCUse'] = temp_df['HCUse'].replace({0: 'Yes', 1: 'No'})

fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
plt.tight_layout(h_pad=5)

q = sns.lineplot(x='cycle_day', y='E2', data=temp_df, hue='subject', 
                 hue_order=['sub-01', 'sub-02', 'sub-03', 'sub-04'], 
                 style='HCUse', style_order=['Yes', 'No'],
                 palette=ppt_pal, ax=ax[0], legend=False, 
                 #markers='.', 
                 markersize=12, linewidth=1.5)

q.set_xlabel('Cycle day', fontsize=20)
q.set_ylabel('[Estradiol], scaled', fontsize=20)
q.tick_params(axis='both', which='major', labelsize=20)
q.tick_params(axis='both', which='minor', labelsize=12)

r = sns.lineplot(x='cycle_day', y='P4', data=temp_df, hue='subject',
                 hue_order=['sub-01', 'sub-02', 'sub-03', 'sub-04'], 
                 style='HCUse', style_order=['Yes', 'No'],
                 palette=ppt_pal, ax=ax[1], #markers='.', 
                 markersize=12, linewidth=1.5)

r.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
         prop={'size': 20})
r.set_xlabel('Cycle day', fontsize=20)
r.set_ylabel('[Progesterone], scaled', fontsize=20)
r.tick_params(axis='both', which='major', labelsize=20)
r.tick_params(axis='both', which='minor', labelsize=12)

sns.despine()
fig.savefig('estradiol_progesterone_levels.png', dpi=400, bbox_inches='tight')