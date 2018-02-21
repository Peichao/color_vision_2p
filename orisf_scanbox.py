drive = 'I'
animal = 'AE5'
unit = '006'
trial = '006'

file_path = '%s:\\AE5\\%s_%s_%s.signals' % (drive, animal, unit, trial)
analyzer_path = 'E:\\AE5\\AnalyzerFiles\\%s_u%s_%s.analyzer' % (animal, unit, trial)
info_file = "%s:\\AE5\\%s_%s_%s.mat" % (drive, animal, unit, trial)
ref_path = "%s:\\AE5\\%s_%s_%s.ref" % (drive, animal, unit, trial)
segment_file = "%s:\\AE5\\%s_%s_%s.segment" % (drive, animal, unit, trial)
plot = 'y'
plot_gaussian_fit = 'n'

import scipy.io as sio
import functions

F = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)['sig']
info = sio.loadmat(info_file, squeeze_me=True, struct_as_record=False)['info']
segment = sio.loadmat(segment_file, squeeze_me=True, struct_as_record=False)['vert']
ref = sio.loadmat(ref_path, squeeze_me=True, struct_as_record=False)['ref']
params = functions.analyzer_params(analyzer_path)
trial_num, stim_time = functions.analyzer_pg_conds(analyzer_path)
trial_num['direction'] = trial_num.ori
trial_num.ori[(trial_num.ori >= 180)] = trial_num.ori[(trial_num.ori >= 180)] - 180

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats

start_frames = info.frame[0::2]
end_frames = info.frame[1::2]
trial_length = np.min(end_frames - start_frames)

mean_image = ref
roi_mask = np.zeros(ref.shape, dtype=bool)
ori_mask = np.zeros(ref.shape)
ori_mask[:, :] = -1

ori_info = pd.DataFrame(columns=['ori_pref', 'osi'])
pref_cells = pd.DataFrame(columns=['expt', 'x', 'y', 'ori_pref'])

num_cells = np.size(segment)
for i in np.arange(num_cells):
    if (np.size(segment[i]) > 2) & (np.isnan(np.max(F[:, i])) == False):
        segment[i] = segment[i].astype(int)
        roi_mask[(segment[i][:, 1] - 1), (segment[i][:, 0] - 1)] = True

        Fcell_split = np.split(F[:, i], info.frame)[1::2]
        Fcell = np.zeros([trial_num.shape[0], trial_length])
        for j, cell in enumerate(Fcell_split):
            Fcell[j, :] = cell[0:120]

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
        cell_F_stim.insert(1, 'Spatial Frequency', trial_num.s_freq)

        # Remove blanks (for now)
        cell_F_stim = cell_F_stim[cell_F_stim['Spatial Frequency'] != 256]

        counts_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).mean()
        counts_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).mean()
        sem_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).sem()
        sem_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).sem()

        s_freq_max = counts_ori.unstack(level=1).mean(axis=0).idxmax()
        orientation_max = counts_ori.unstack(level=1).mean(axis=1).idxmax()

        # Use vector averaging to calculate the preferred orientation (Ohki et al., 2005)
        ori = np.deg2rad(counts_sf.Mean[s_freq_max[1]].index.values)
        # responses = counts_sf.Mean[s_freq_max[1]].values
        # a = np.sum(responses * np.cos(2 * ori))
        # b = np.sum(responses * np.sin(2 * ori))
        # ori_pref_vector = np.rad2deg(0.5 * np.arctan(b / a))

        # F_stim_sf_max = cell_F_stim[cell_F_stim['Spatial Frequency'] == s_freq_max[1]]
        # list_F_stim = [F_stim_sf_max[F_stim_sf_max['Orientation'] == deg]['Mean'] for deg in np.rad2deg(ori)]
        list_F_stim = [cell_F_stim[cell_F_stim['Orientation'] == deg]['Mean'] for deg in np.rad2deg(ori)]
        F_value, p = sp.stats.f_oneway(*list_F_stim)

        if plot_gaussian_fit == 'y':
            # fit gaussian curve to orientation curve, calculate preferred orientation and OSI
            x = counts_sf['Mean'][s_freq_max[1]].index.values
            y = counts_sf['Mean'][s_freq_max[1]].values.T

            x_new = np.linspace(x.min(), x.max(), 100)

            n = len(x)
            mean = sum(x * y) / n
            sigma = sum(y * (x - mean) ** 2) / n

            from scipy.optimize import curve_fit

            popt, pcov = curve_fit(functions.gaus, x, y, p0=[1, mean, sigma], maxfev=1000000)
            ori_pref = x_new[np.argmax(functions.gaus(x_new, *popt))]
            r_pref = functions.gaus(ori_pref, *popt)
            r_orth = functions.gaus(ori_pref + 90, *popt)
            osi = (r_pref - r_orth) / r_pref

        else:
            ori_pref = counts_sf['Mean'][s_freq_max[1]].idxmax()
            r_pref = counts_sf['Mean'][s_freq_max[1]].loc[ori_pref]
            if ori_pref < 90:
                ori_orth = ori_pref + 90
            else:
                ori_orth = ori_pref - 90
            r_orth = counts_sf['Mean'][s_freq_max[1]].loc[ori_orth]
            osi = (r_pref - r_orth) / r_pref

        if p < 0.01:
            if osi > 0.3:
                ori_mask[(segment[i][:, 1] - 1), (segment[i][:, 0] - 1)] = ori_pref

                if plot == 'y':
                    plt.ioff()
                    fig, (ax1, ax2) = plt.subplots(2, 1)
                    colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

                    counts_ori['Mean'][orientation_max].plot(ax=ax1, linewidth=2, legend=False)
                    ax1.fill_between(sem_ori['Mean'][orientation_max].index.values,
                                     counts_ori['Mean'][orientation_max].values - sem_ori['Mean'][
                                         orientation_max].values,
                                     counts_ori['Mean'][orientation_max].values + sem_ori['Mean'][
                                         orientation_max].values,
                                     alpha=0.5)
                    ax1.set_title('Spatial Frequency Tuning')
                    ax1.set_ylabel(r'Response ($\Delta F/F$)')
                    ax1.set_xlabel('Spatial Frequency (cycles/degree)')

                    counts_sf['Mean'][s_freq_max[1]].plot(ax=ax2, linewidth=2, legend=False)
                    ax2.fill_between(sem_sf['Mean'][s_freq_max[1]].index.values,
                                     counts_sf['Mean'][s_freq_max[1]].values - sem_sf['Mean'][s_freq_max[1]].values,
                                     counts_sf['Mean'][s_freq_max[1]].values + sem_sf['Mean'][s_freq_max[1]].values,
                                     alpha=0.5)
                    if plot_gaussian_fit == 'y':
                        ax2.plot(x_new, functions.gaus(x_new, *popt), 'r--', label='fit')

                    ax2.set_title('Orientation Tuning')
                    ax2.set_ylabel(r'Response ($\Delta F/F$)')
                    ax2.set_xlabel('Orientation (degrees)')
                    # plt.suptitle('Pref. Orientation: %.2f, OSI: %.2f' % (ori_pref, osi))
                    plt.tight_layout()

                    plt.savefig('E:/test/cell%d.pdf' % i, format='pdf', bbox_inches='tight')
                    plt.close()
                    plt.ion()

                # x_med, y_med = cell.med
                # pref_cells.loc[i] = [analyzer_path[-17:-9], x_med, y_med, ori_pref]
                # pref_cells.to_csv('g:/test/%s_ori_pref_df.csv' % analyzer_path[-17:-9])
        ori_info.loc[i] = np.array([ori_pref, osi])

from matplotlib.colors import ColorConverter

plt.figure()
cmap = plt.cm.get_cmap('plasma')
cmap.set_under(ColorConverter.to_rgba('white', alpha=0))
plt.imshow(ori_mask, cmap=cmap, vmin=0)
plt.colorbar()
plt.title('Orientation Tuning Map', fontsize=24)
plt.show()
