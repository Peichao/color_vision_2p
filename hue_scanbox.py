drive = 'I'
animal = 'AE5'
unit = '006'
trial = '007'

file_path = '%s:\\AE5\\%s_%s_%s.signals' % (drive, animal, unit, trial)
analyzer_path = 'E:\\AE5\\AnalyzerFiles\\%s_u%s_%s.analyzer' % (animal, unit, trial)
info_file = "%s:\\AE5\\%s_%s_%s.mat" % (drive, animal, unit, trial)
ref_path = "%s:\\AE5\\%s_%s_%s.ref" % (drive, animal, unit, trial)
align_path = "%s:\\AE5\\%s_%s_%s.align" % (drive, animal, unit, trial)
segment_file = "%s:\\AE5\\%s_%s_%s.segment" % (drive, animal, unit, trial)
plot = 'y'
plot_gaussian_fit = 'n'

color_hex = ['#af1600', '#8a4600', '#5a5d01', '#2a6600', '#006a00', '#006931', '#006464', '#0058b6', '#002DFF', '#6a2ade', '#97209b', '#aa1c50']

import scipy.io as sio
import functions

F = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sig']
info = sio.loadmat(info_file, squeeze_me=True, struct_as_record=False)['info']
segment = sio.loadmat(segment_file, squeeze_me=True, struct_as_record=False)['vert']
ref = sio.loadmat(ref_path, squeeze_me=True, struct_as_record=False)['ref']
align = sio.loadmat(align_path, squeeze_me=True, struct_as_record=False)['m']
params = functions.analyzer_params(analyzer_path)
trial_num, stim_time = functions.analyzer_pg_conds(analyzer_path)
trial_num['direction'] = trial_num.ori
trial_num.ori[(trial_num.ori >= 180)] = trial_num.ori[(trial_num.ori >= 180)] - 180

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

info.frame = info.frame.astype(int)
frame_diff = np.diff(info.frame)
frame_reset = np.where(frame_diff < 0)
for reset in frame_reset[0]:
    info.frame[(reset+1):] += 65536
start_frames = info.frame[0::2]
end_frames = info.frame[1::2]
trial_length = np.min(end_frames - start_frames)

mean_image = ref
roi_mask = np.zeros(ref.shape, dtype=bool)
color_mask = np.zeros(ref.shape)

ori_info = pd.DataFrame(columns=['ori_pref', 'osi'])
pref_cells = pd.DataFrame(columns=['expt', 'x', 'y', 'ori_pref'])

num_cells = np.size(segment)
p_list = []
csi_list = []

for i in np.arange(num_cells):
    if (np.size(segment[i]) > 2) & (np.isnan(np.max(F[:, i])) == False):
        segment[i] = segment[i].astype(int)
        roi_mask[(segment[i][:, 1] - 2), (segment[i][:, 0] - 2)] = True

        Fcell_split = np.split(F[:, i], info.frame)[1::2]
        Fcell = np.zeros([trial_num.shape[0], trial_length])
        for j, cell in enumerate(Fcell_split):
            Fcell[j, :] = cell[0:trial_length]

        t = np.linspace(stim_time[0] * -1, stim_time.sum() - stim_time[1], trial_length)
        cell_F = pd.DataFrame(Fcell)
        cell_F.insert(0, 'Orientation', trial_num.ori)
        cell_F.insert(1, 'Spatial Frequency', trial_num.s_freq)
        # cell_F.insert(2, 'Baseline',
        #               np.reshape(Fcell[i, :], (int(Fcell.shape[1] / trial_length), trial_length))[:,
        #               0:int(np.floor(trial_length * stim_time[0] / stim_time.sum()))].mean(axis=1))
        pre_stim_samples = trial_length * (stim_time[0] / stim_time.sum())
        baseline = Fcell[:, 5:int(pre_stim_samples)].mean(axis=1)
        cell_F.insert(2, 'Baseline', baseline)

        start_sample = int(np.ceil(trial_length * (stim_time[0] + 0.5) / stim_time.sum()))
        end_sample = int(trial_length - np.ceil(trial_length * (stim_time[1]) / stim_time.sum()))

        cell_F.iloc[:, trial_length*-1:] = cell_F.iloc[:, trial_length*-1:].subtract(cell_F['Baseline'], axis=0).divide(
            cell_F['Baseline'],
            axis=0)

        cell_F_stim = pd.DataFrame(cell_F.iloc[:, -(trial_length - start_sample):-(trial_length - end_sample)].mean(axis=1),
                                   columns=['Mean'])

        cell_F_stim.insert(0, 'Orientation', trial_num.ori)
        cell_F_stim.insert(1, 'Direction', trial_num.direction)
        cell_F_stim.insert(2, 'Spatial Frequency', trial_num.s_freq)
        cell_F_stim.insert(3, 'Color', trial_num.colormod)

        # Remove blanks (for now)
        cell_F_stim = cell_F_stim[cell_F_stim.Color != 34]
        cell_F_stim = cell_F_stim[cell_F_stim['Spatial Frequency'] != 256]

        max_response = cell_F_stim.groupby('Color').mean()['Mean'].max()

        # Calculate OSI and save into DataFrame
        counts_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).mean()
        counts_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).mean()
        sem_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).sem()
        sem_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).sem()

        s_freq_max = counts_ori.unstack(level=1).mean(axis=0).idxmax()
        orientation_max = counts_ori.unstack(level=1).mean(axis=1).idxmax()
        ori_pref = counts_sf['Mean'][s_freq_max[1]].idxmax()
        r_pref = counts_sf['Mean'][s_freq_max[1]].loc[ori_pref]
        if ori_pref < 90:
            ori_orth = ori_pref + 90
        else:
            ori_orth = ori_pref - 90
        r_orth = counts_sf['Mean'][s_freq_max[1]].loc[ori_orth]
        osi = (r_pref - r_orth) / r_pref

        colors = np.unique(trial_num.colormod.values)
        colors = np.delete(colors, np.where(colors == 256))
        colors = np.delete(colors, np.where(colors == 34))
        list_F_stim = [cell_F_stim[cell_F_stim['Color'] == color]['Mean'] for color in colors]
        F_value, p = sp.stats.f_oneway(*list_F_stim)
        p_list.append(p)

        if p < 0.01:
            max_response = cell_F_stim.groupby('Color').mean()['Mean'].max()
            if max_response > 0.1:
                max_color = cell_F_stim.groupby('Color').mean()['Mean'].idxmax()
                if max_color < 19:
                    ortho_color = max_color + 6
                else:
                    ortho_color = max_color - 6
                ortho_response = cell_F_stim.groupby('Color').mean()['Mean'][ortho_color]
                color_si = (max_response - ortho_response) / max_response
                csi_list.append(color_si)
                if color_si > 0.7:
                    color_mask[(segment[i][:, 1] - 2), (segment[i][:, 0] - 2)] = max_color
                    if plot == 'y':
                        plt.ioff()
                        fig, ax = plt.subplots()
                        ax.scatter(cell_F_stim.groupby('Color').mean()['Mean'].index.values,
                                   cell_F_stim.groupby('Color').mean()['Mean'].values, color=color_hex, s=100, zorder=2)

                        ax.fill_between(cell_F_stim.groupby('Color').mean()['Mean'].index.values,
                                        cell_F_stim.groupby('Color').mean()['Mean'] -
                                        cell_F_stim.groupby('Color').sem()['Mean'],
                                        cell_F_stim.groupby('Color').mean()['Mean'] +
                                        cell_F_stim.groupby('Color').sem()['Mean'],
                                        alpha=0.5, color='#808080', zorder=1)
                        ax.xaxis.set_ticks([])
                        ax.set_ylabel(r'Response ($\Delta F/F$)')
                        plt.savefig('E:/AE5/cells/cell%d.pdf' % i, format='pdf')
                        plt.close()

from matplotlib.colors import ListedColormap
from matplotlib.colors import ColorConverter
cbar_colors = ['#af1600', '#8a4600', '#5a5d01', '#2a6600', '#006a00', '#006931', '#006464', '#0058b6', '#002DFF', '#6a2ade', '#97209b', '#aa1c50']
cmap = ListedColormap(cbar_colors, 'indexed')
cmap.set_under(ColorConverter.to_rgba('white', alpha=0))

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(align, cmap='gray')
im = ax.imshow(color_mask.astype(int), cmap=cmap, vmin=13, vmax=24)
np.save('E:/AE5/cells/%s_color_map.npy' % analyzer_path[-17:-9], color_mask)
cb = fig.colorbar(im)
cb.set_ticks([])
cb.set_label('Preferred Color')
plt.axis('off')
plt.show()
plt.close()

import glob
color_map_files = glob.glob('E:\\AE5\\cells\\*color_map.npy')
fig, ax = plt.subplots()
for file in color_map_files:
    color_map = np.load(file)
    ax.imshow(color_map, cmap=cmap, vmin=13, vmax=24)
ax.set_xticks([])
ax.set_yticks([])
cb = fig.colorbar(im)
cb.set_ticks([])
cb.set_label('Preferred Color')
plt.title('Color Tuning Map')
plt.savefig('E:/AE5/cells/color_map_all.pdf', format='pdf', bbox_inches='tight')
plt.show()
