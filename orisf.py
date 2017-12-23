import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import functions

file_path = 'D:/2P_data/F/AE1/2017-06-22/F_AE1_2017-06-22_plane1_proc.mat'
analyzer_path = 'F:/NHP/AE1/AnalyzerFiles/AE1_u004_011.analyzer'
plot = 'y'

F = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)['dat']
trial_num, stim_time = functions.analyzer_pg(analyzer_path)
trial_num['direction'] = trial_num.ori
trial_num.ori[(trial_num.ori >= 180) & (trial_num.ori != 256)] = trial_num.ori[(trial_num.ori >= 180) &
                                                                               (trial_num.ori != 256)] - 180

mean_image = F.mimg[:, :, 1]
roi_mask = np.zeros(mean_image.shape, dtype=bool)
ori_mask = np.zeros(mean_image.shape)
ori_mask[:, :] = np.nan

pref_cells = pd.DataFrame(columns=['expt', 'x', 'y', 'color_pref'])
ori_info = pd.DataFrame(columns=['ori_pref', 'osi'])

for i, cell in enumerate(F.stat):
    print('Analyzing cell %d of %d.' % (i, len(F.stat)))
    if cell.iscell == 1:
        roi_mask[(cell.ypix-1), (cell.xpix-1)] = True

        scans = int(F.Fcell.shape[1] / trial_num.shape[0])
        t = np.linspace(stim_time[0] * -1, stim_time.sum() - stim_time[1], scans)
        cell_F = pd.DataFrame(np.reshape(F.Fcell[i, :], (int(F.Fcell.shape[1] / scans), scans)))
        cell_F.insert(0, 'Orientation', trial_num.ori)
        cell_F.insert(1, 'Spatial Frequency', trial_num.s_freq)
        cell_F.insert(2, 'Baseline',
                      np.reshape(F.Fcell[i, :], (int(F.Fcell.shape[1] / scans), scans))[:, 0:int(np.floor(scans * stim_time[0] / stim_time.sum()))].mean(axis=1))

        start_sample = int(np.ceil(scans * (stim_time[0] + 0.5) / stim_time.sum()))
        end_sample = int(scans - np.ceil(scans * (stim_time[1]) / stim_time.sum()))

        cell_F.ix[:, -scans:] = cell_F.ix[:, -scans:].subtract(cell_F['Baseline'], axis=0).divide(cell_F['Baseline'],
                                                                                                  axis=0)

        cell_F_stim = pd.DataFrame(cell_F.ix[:, -(scans-start_sample):-(scans-end_sample)].mean(axis=1), columns=['Mean'])
        cell_F_stim.insert(0, 'Orientation', trial_num.ori)
        cell_F_stim.insert(1, 'Spatial Frequency', trial_num.s_freq)
        counts_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).mean()
        counts_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).mean()
        sem_ori = cell_F_stim.groupby(['Orientation', 'Spatial Frequency']).sem()
        sem_sf = cell_F_stim.groupby(['Spatial Frequency', 'Orientation']).sem()

        s_freq_max = counts_ori.drop(256).unstack(level=1).mean(axis=0).idxmax()
        orientation_max = counts_ori.drop(256).unstack(level=1).mean(axis=1).idxmax()

        # fit gaussian curve to orientation curve, calculate preferred orientation and OSI
        x = counts_sf.drop(256).ix[s_freq_max[1]].index.values
        y = counts_sf.drop(256).ix[s_freq_max[1]].values.T[0]

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

        if osi > 0.7:
            ori_mask[(cell.ypix - 1), (cell.xpix - 1)] = ori_pref
        ori_info.loc[i] = np.array([ori_pref, osi])

        if plot == 'y':
            plt.ioff()
            fig, (ax1, ax2) = plt.subplots(2, 1)
            colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

            counts_ori.drop(256).ix[orientation_max].plot(ax=ax1, linewidth=2, legend=False)
            ax1.set_title('Spatial Frequency Tuning')
            ax1.set_ylabel('Percent Change')
            ax1.set_xlabel('Spatial Frequency (cycles/degree)')

            counts_sf.drop(256).ix[s_freq_max[1]].plot(ax=ax2, linewidth=2, legend=False)
            ax2.plot(x_new, functions.gaus(x_new, *popt), 'r--', label='fit')

            ax2.set_title('Orientation Tuning')
            ax2.set_ylabel('Percent Change')
            ax2.set_xlabel('Orientation (degrees)')

            fig.tight_layout()
            plt.suptitle('Pref. Orientation: %.2f, OSI: %.2f' % (ori_pref, osi))
            plt.savefig('D:/2P_data/cell' + str(i) + '.pdf', format='pdf')
            plt.clf()
            plt.close()

cell_mask = np.ma.masked_where(~roi_mask, np.ones(mean_image.shape))
plt.imshow(mean_image, cmap='gray')
plt.imshow(cell_mask, cmap='bwr', alpha=0.4, vmin=0, vmax=1)
plt.show()

plt.imshow(ori_mask, cmap='viridis')
plt.colorbar()
plt.title('Orientation Tuning Map', fontsize=24)
plt.show()
