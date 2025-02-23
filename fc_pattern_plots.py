import json
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, datasets
from os.path import join

sns.set(style='ticks', context='paper', font_scale=2)
#plt.rcParams["font.family"] = "Avenir"
#plt.rcParams["font.size"] = 20
#plt.rcParams['font.monospace'] = 'Courier New'


# Opening JSON file with paths in it
f = open('paths.json') 

# returns JSON object as a dictionary 
data = json.load(f) 

craddock_atlas = data['craddock_fname']
TRAIN_DIR = data['TRAIN_DIR']

coords = plotting.find_parcellation_cut_coords(craddock_atlas)

colors = pd.read_csv(data['network_colors'], 
                     index_col=0, dtype=str)

ntwk_colors = sns.color_palette(colors['0'].values)
network_labels = pd.read_csv('craddock_regions_to_networks.csv', header=0, index_col=0, dtype=str)
for i in range(270):
    if i not in network_labels.index:
        print(i)

network_labels.index = range(1,269)
sorted_ = network_labels.sort_values('0').index.astype(int)
sort_idx = sorted_.values - 1
sorted_networks = network_labels.sort_values('0')
for i in range(1,len(sorted_networks.index)):
    if sorted_networks.iloc[i].values[0] not in sorted_networks.iloc[:i].values:
        pass
    else:
        sorted_networks.iloc[i] = ''

#final results for bc (including ovulatory window)
bc_path = join(
    TRAIN_DIR, 
    'nbs-predict_outcome-bc_confounds-framewise_displacement_weighted-08_05_2023.tsv'
)
e2_path = join(
    TRAIN_DIR, 
    'nbs-predict_outcome-estradiol_confounds-[\'framewise_displacement\']_weighted-08_03_2023.tsv')
e2bc_path = join(
    TRAIN_DIR, 
    'nbs-predict_outcome-estradiol_confounds-[\'framewise_displacement\', \'bc\']_weighted-07_30_2023.tsv')
# read in p4 features
p4_path = join(
    TRAIN_DIR, 
    'nbs-predict_outcome-progesterone_confounds-[\'framewise_displacement\']_weighted-08_05_2023.tsv')
p4bc_path = join(
    TRAIN_DIR, 
    'nbs-predict_outcome-progesterone_confounds-[\'framewise_displacement\', \'bc\']_weighted-08_03_2023.tsv')

bc_weights = pd.read_table(bc_path, index_col=0, header=0)
bc_strength = np.sum(bc_weights, axis=1)
bc_strength = bc_strength / bc_strength.max() * 20

e2_weights = pd.read_table(e2_path, index_col=0, header=0)
e2bc_weights = pd.read_table(e2bc_path, index_col=0, header=0)
e2_pos = np.where(e2_weights.values > 0, e2_weights.values, 0)
e2bc_pos = np.where(e2bc_weights.values > 0, e2bc_weights.values, 0)
e2_strength = e2_pos.sum(axis=1) / e2_pos.sum(axis=1).max() * 60
e2bc_strength = e2bc_pos.sum(axis=1) / e2bc_pos.sum(axis=1).max() * 60

p4_weights = pd.read_table(p4_path, index_col=0, header=0)
p4bc_weights = pd.read_table(p4bc_path, index_col=0, header=0)
p4_pos = np.where(p4_weights > 0, p4_weights, 0)
p4bc_pos = np.where(p4bc_weights > 0, p4bc_weights, 0)
p4_strength = p4_pos.sum(axis=1) / p4_pos.sum(axis=1).max() * 60
p4bc_strength = p4bc_pos.sum(axis=1) / p4bc_pos.sum(axis=1).max() * 60

# plot glass brain connectomes, with color-coded nodes and edges
# node size proportional to node strength

g = plotting.plot_connectome(bc_weights, 
                         coords, 
                         edge_cmap='seismic',
                         edge_kwargs={'alpha': 0.3},
                         edge_threshold='99.99999999%', 
                         colorbar=True,
                         annotate=False, 
                         display_mode='lyrz',
                         node_size=bc_strength,
                         node_color=colors['0'])
g.savefig('../figures/bc_color-coded.png', dpi=400)

g = plotting.plot_connectome(e2_weights, 
                         coords, 
                         node_size=e2_strength,
                         edge_cmap='seismic',
                         edge_kwargs={'alpha': 0.4},
                         edge_threshold='99.9%', 
                         colorbar=True, 
                         display_mode='lyrz',
                         node_color=colors['0'],
                         annotate=False,
                         #node_color='auto'
                        )
g.savefig('../figures/e2_color-coded.png', dpi=400)

g = plotting.plot_connectome(e2bc_pos, 
                         coords, 
                         node_size=e2bc_strength,
                         edge_cmap='seismic',
                         edge_kwargs={'alpha': 0.4},
                         edge_threshold='99.9%', 
                         colorbar=True, 
                         annotate=False,
                         display_mode='lyrz',
                         node_color=colors['0'],
                         #node_color='auto'
                        )
g.savefig('../figures/e2bc_color-coded.png', dpi=400)

g = plotting.plot_connectome(p4_weights, 
                         coords, 
                         node_size=p4_strength,
                         edge_cmap='seismic',
                         edge_kwargs={'alpha': 0.4},
                         display_mode='lyrz',
                         edge_threshold='99.9%', 
                         colorbar=True, 
                         node_color=colors['0'],
                         annotate=False,
                         #node_color='auto'
                        )
g.savefig('../figures/p4_color-coded.png', dpi=400)

g = plotting.plot_connectome(p4bc_pos, 
                         coords, 
                         node_size=p4bc_strength,
                         edge_cmap='seismic',
                         edge_kwargs={'alpha': 0.4},
                         display_mode='lyrz',
                         edge_threshold='99.9%', 
                         colorbar=True, 
                         node_color=colors['0'],
                         annotate=False,
                         #node_color='auto'
                        )
g.savefig('../figures/p4bc_color-coded.png', dpi=400)

# now we wrangle the data for importance matrices

bc_temp = bc_weights.loc[sort_idx]
bc_ntwk = bc_temp.T.iloc[sort_idx]
bc_ntwk.index = list(network_labels.sort_values('0').values.flatten())
bc_ntwk.columns = list(network_labels.sort_values('0').values.flatten())

e2_pos_df = pd.DataFrame(e2_pos, columns=e2_weights.columns, index=e2_weights.index)
e2bc_pos_df = pd.DataFrame(e2bc_pos, columns=e2bc_weights.columns, index=e2bc_weights.index)

e2_temp = e2_pos_df.loc[sort_idx]
e2_ntwk = e2_temp.T.iloc[sort_idx]
e2_ntwk.index = list(network_labels.sort_values('0').values.flatten())
e2_ntwk.columns = list(network_labels.sort_values('0').values.flatten())

e2bc_temp = e2bc_pos_df.loc[sort_idx]
e2bc_ntwk = e2bc_temp.T.iloc[sort_idx]
e2bc_ntwk.index = list(network_labels.sort_values('0').values.flatten())
e2bc_ntwk.columns = list(network_labels.sort_values('0').values.flatten())


p4_pos_df = pd.DataFrame(p4_pos, columns=p4_weights.columns, index=p4_weights.index)
p4bc_pos_df = pd.DataFrame(p4bc_pos, columns=p4bc_weights.columns, index=p4bc_weights.index)

p4_temp = p4_pos_df.loc[sort_idx]
p4_ntwk = p4_temp.T.iloc[sort_idx]
p4_ntwk.index = list(network_labels.sort_values('0').values.flatten())
p4_ntwk.columns = list(network_labels.sort_values('0').values.flatten())

p4bc_temp = p4bc_pos_df.loc[sort_idx]
p4bc_ntwk = p4bc_temp.T.iloc[sort_idx]
p4bc_ntwk.index = list(network_labels.sort_values('0').values.flatten())
p4bc_ntwk.columns = list(network_labels.sort_values('0').values.flatten())

# plotting importance matrices as heatmaps
fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(bc_ntwk, 
            square=True, 
            cmap='seismic', 
            center=0,
            #xticklabels=10, 
            #yticklabels=10,  
            linewidths=0, ax=ax)
ax.tick_params(left=False, bottom=False)
fig.savefig('../figures/bc_ordered_heatmap.png', dpi=400)

fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(e2_ntwk, 
            square=True, 
            cmap='seismic', 
            center=0,
            xticklabels=10, 
            yticklabels=10,  
            linewidths=0, ax=ax)
ax.tick_params(left=False, bottom=False)
fig.savefig('../figures/e2_ordered_heatmap.png', dpi=400)

fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(e2bc_ntwk, 
            square=True, 
            cmap='seismic', 
            center=0,
            xticklabels=10, 
            yticklabels=10,  
            linewidths=0, 
            ax=ax)
ax.set_yticks(range(0,268))
ax.set_yticklabels(sorted_networks['0'])
ax.tick_params(left=False, bottom=False)
fig.savefig('../figures/e2bc_ordered_heatmap.png', dpi=400)

fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(p4_ntwk, 
            square=True, 
            cmap='seismic', 
            center=0,
            xticklabels=10, 
            yticklabels=10,  
            linewidths=0, ax=ax)

ax.set_yticklabels('')
ax.set_xticklabels('')
ax.tick_params(left=False, bottom=False)
fig.savefig('../figures/p4_ordered_heatmap.png', dpi=400)

fig,ax = plt.subplots(figsize=(4,3))
sns.heatmap(p4bc_ntwk, 
            square=True, 
            cmap='seismic', 
            center=0,
            xticklabels=1, 
            yticklabels=10,  
            vmin=0.,
            linewidths=0, ax=ax)

ticklist = ax.get_xticklabels()
region_order = list(p4_temp.index)

for ticklabel in ax.get_xticklabels():
    i = ticklist.index(ticklabel)
    ticklabel.set_color(colors['0'].iloc[region_order[i]])
ax.set
ax.tick_params(left=False, bottom=False)
fig.savefig('../figures/p4bc_ordered_heatmap.png', dpi=400)

# now we wrangle for the supplemental maps
# summarizing feature importance at the network level
p4_mean_max = pd.DataFrame(
    index=p4_ntwk.index.unique(), 
    columns=p4_ntwk.index.unique(),
    dtype=float
)
for network in p4_ntwk.index.unique():
    for network2 in p4_ntwk.index.unique():
        p4_mean_max.at[network, network2] = np.mean(np.mean(p4_ntwk.loc[network][network2]))

p4bc_mean_max = pd.DataFrame(
    index=p4bc_ntwk.index.unique(), 
    columns=p4bc_ntwk.index.unique(),
    dtype=float
)
for network in p4bc_ntwk.index.unique():
    for network2 in p4bc_ntwk.index.unique():
        p4bc_mean_max.at[network, network2] = np.mean(np.mean(p4bc_ntwk.loc[network][network2]))
        
e2_mean_max = pd.DataFrame(
    index=e2_ntwk.index.unique(), 
    columns=e2_ntwk.index.unique(),
    dtype=float
)
for network in e2_ntwk.index.unique():
    for network2 in e2_ntwk.index.unique():
        e2_mean_max.at[network, network2] = np.mean(np.mean(e2_ntwk.loc[network][network2]))
e2bc_mean_max = pd.DataFrame(
    index=e2bc_ntwk.index.unique(), 
    columns=e2bc_ntwk.index.unique(),
    dtype=float
)
for network in e2bc_ntwk.index.unique():
    for network2 in e2bc_ntwk.index.unique():
        e2bc_mean_max.at[network, network2] = np.mean(np.mean(e2bc_ntwk.loc[network][network2]))      
bc_mean_min = pd.DataFrame(
    index=bc_ntwk.index.unique(), 
    columns=bc_ntwk.index.unique(),
    dtype=float
)
for network in bc_ntwk.index.unique():
    for network2 in bc_ntwk.index.unique():
        bc_mean_min.at[network, network2] = np.mean(np.min(bc_ntwk.loc[network][network2]))
        
bc_mean_max = pd.DataFrame(
    index=bc_ntwk.index.unique(), 
    columns=bc_ntwk.index.unique(),
    dtype=float
)
for network in bc_ntwk.index.unique():
    for network2 in bc_ntwk.index.unique():
        bc_mean_max.at[network, network2] = np.mean(np.mean(bc_ntwk.loc[network][network2]))

# and plotting these importance matrices as heatmaps
sns.set(font_scale=1)
fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(10,13), layout='constrained')

sns.heatmap(
    bc_mean_max, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[0,0]
)
ax[0,0].set_title('HC use: Mean')
sns.heatmap(
    bc_mean_min, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[0,1]
)
ax[0,1].set_title('HC use: Min')
# estradiol
sns.heatmap(
    e2_mean_max, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[1,0]
)
ax[1,0].set_title('Estradiol')
sns.heatmap(
    e2bc_mean_max, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[1,1]
)
ax[1,1].set_title('(controlling for HC use)')

sns.heatmap(
    p4_mean_max, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[2,0]
)
ax[2,0].set_title('Progesterone')
sns.heatmap(
    p4bc_mean_max, 
    square=True, 
    cmap='seismic', 
    center=0,
    ax=ax[2,1]
)
ax[2,1].set_title('(controlling for HC use)')
fig.savefig('../figures/network_wise_maxes.png', bbox_inches='tight', dpi=600)

# and now we wrangle for node strength slices
nimg = nib.load(craddock_atlas)
cmap = 'seismic'

e2_strength = e2_pos_df.sum()
e2_strength.index = colors.index

regn_sch_arr = nimg.get_fdata()
for i in colors.index:
    regn_sch_arr[np.where(regn_sch_arr == i)] = np.sum(e2_strength.loc[i])
strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
nib.save(strength_nimg, 'e2_strength.nii')

# and plot!
sns.set_context('paper', font_scale=2)
fig = plotting.plot_stat_map(strength_nimg, 
                       cmap=cmap, 
                       colorbar=True, 
                       display_mode='z',
                       annotate=True, 
                       threshold=0.5, 
                       symmetric_cbar=True 
                       )
fig.savefig('../figures/e2_strength_slices1.png', dpi=400)

e2bc_strength = e2bc_pos_df.sum()
e2bc_strength.index = colors.index

fsaverage = datasets.fetch_surf_fsaverage()
nimg = nib.load(craddock_atlas)
regn_sch_arr = nimg.get_fdata()
for i in colors.index:
    regn_sch_arr[np.where(regn_sch_arr == i)] = np.sum(e2bc_strength.loc[i])
strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
# replace this filename with BIDSy output
#nib.save(strength_nimg, f'/Users/katherine.b/Dropbox/{title}predictive-strength.nii')
fig = plotting.plot_stat_map(strength_nimg, 
                       cmap=cmap, 
                       colorbar=True, 
                       display_mode='z',
                       annotate=True, 
                       threshold=0.1, 
                       symmetric_cbar=True 
                       )
fig.savefig('../figures/e2bc_strength_slices1.png', dpi=400)
#nib.save(strength_nimg, 'e2bc_strength.nii')

p4_strength = p4_pos_df.sum()
p4_strength.index = colors.index

fsaverage = datasets.fetch_surf_fsaverage()
nimg = nib.load(craddock_atlas)
regn_sch_arr = nimg.get_fdata()
for i in colors.index:
    regn_sch_arr[np.where(regn_sch_arr == i)] = np.sum(p4_strength.loc[i])
strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
# replace this filename with BIDSy output
#nib.save(strength_nimg, f'/Users/katherine.b/Dropbox/{title}predictive-strength.nii')
fig = plotting.plot_stat_map(strength_nimg, 
                       cmap=cmap, 
                       colorbar=True, 
                       display_mode='z',
                       annotate=True, 
                       threshold=0.5, 
                       symmetric_cbar=True
                       )
fig.savefig('../figures/p4_strength_slices.png', dpi=400)
nib.save(strength_nimg, 'p4_strength.nii')

p4bc_strength = p4bc_pos_df.sum()
p4bc_strength.index = colors.index

fsaverage = datasets.fetch_surf_fsaverage()
nimg = nib.load(craddock_atlas)
regn_sch_arr = nimg.get_fdata()
for i in colors.index:
    regn_sch_arr[np.where(regn_sch_arr == i)] = np.sum(p4bc_strength.loc[i])
strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
# replace this filename with BIDSy output
#nib.save(strength_nimg, f'/Users/katherine.b/Dropbox/{title}predictive-strength.nii')
fig = plotting.plot_stat_map(strength_nimg, 
                       cmap=cmap, 
                       colorbar=True, 
                       display_mode='z',
                       annotate=True, 
                       threshold=0.1, 
                       symmetric_cbar=True 
                       )
fig.savefig('../figures/p4bc_strength_slices.png', dpi=400)
nib.save(strength_nimg, 'p4bc_strength.nii')

bc_strength = bc_weights.sum()
bc_strength.index = colors.index

fsaverage = datasets.fetch_surf_fsaverage()
nimg = nib.load(craddock_atlas)
regn_sch_arr = nimg.get_fdata()
for i in colors.index:
    regn_sch_arr[np.where(regn_sch_arr == i)] = np.mean(bc_strength.loc[i])
strength_nimg = nib.Nifti1Image(regn_sch_arr, nimg.affine)
# replace this filename with BIDSy output
#nib.save(strength_nimg, f'/Users/katherine.b/Dropbox/{title}predictive-strength.nii')

fig = plotting.plot_stat_map(strength_nimg, 
                       cmap=cmap, 
                       colorbar=True, 
                       display_mode='z',
                       threshold=10,
                       symmetric_cbar=True
                       )
fig.savefig('../figures/bc_strength_slices.png', dpi=400)
nib.save(strength_nimg, 'bc_strength.nii')