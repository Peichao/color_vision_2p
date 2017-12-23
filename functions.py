import pandas as pd
import numpy as np
import scipy.io as sio


def load_analyzer(analyzer_path):
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']
    return analyzer


def load_analyzer(analyzer_path):
    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']
    return analyzer


def analyzer_pg(analyzer_path):
    analyzer = load_analyzer(analyzer_path)

    b_flag = 0
    if type(analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol) != str:
        if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol[0] == 'blank':
            b_flag = 1
    else:
        if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol == 'blank':
            b_flag = 1

    if b_flag == 0:
        if type(analyzer.loops.conds[0].symbol) != str:
            trial_num = np.zeros((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats),
                                 (len(analyzer.loops.conds[0].symbol))))
        else:
            trial_num = np.zeros((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats), 1))
    else:
        if type(analyzer.loops.conds[0].symbol) != str:
            trial_num = np.zeros(((len(analyzer.loops.conds) - b_flag) * len(analyzer.loops.conds[0].repeats) +
                                  len(analyzer.loops.conds[-1].repeats),
                                  (len(analyzer.loops.conds[0].symbol))))
        else:
            trial_num = np.zeros(((len(analyzer.loops.conds) - b_flag) * len(analyzer.loops.conds[0].repeats) +
                                  len(analyzer.loops.conds[-1].repeats), 1))

    for count in range(0, len(analyzer.loops.conds) - b_flag):
        if type(analyzer.loops.conds[count].symbol) != str:
            trial_vals = np.zeros(len(analyzer.loops.conds[count].symbol))
        else:
            trial_vals = np.zeros(1)

        try:
            for count2 in range(0, len(analyzer.loops.conds[count].symbol)):
                trial_vals[count2] = analyzer.loops.conds[count].val[count2]
        except (TypeError, IndexError) as e:
                trial_vals[0] = analyzer.loops.conds[count].val

        for count3 in range(0, len(analyzer.loops.conds[count].repeats)):
            aux_trial = analyzer.loops.conds[count].repeats[count3].trialno
            trial_num[aux_trial - 1, :] = trial_vals

    if b_flag == 1:
        for blank_trial in range(0, len(analyzer.loops.conds[-1].repeats)):
            aux_trial = analyzer.loops.conds[-1].repeats[blank_trial].trialno
            trial_num[aux_trial - 1, :] = np.ones(trial_num.shape[1]) * 256

    stim_time = np.zeros(3)
    for count4 in range(0, 3):
        stim_time[count4] = analyzer.P.param[count4][2]

    return trial_num, stim_time


def analyzer_params(analyzer_path):
    analyzer = load_analyzer(analyzer_path)
    params = {}
    for param in analyzer.P.param:
        params[param[0]] = param[2]

    return params


def analyzer_pg_conds(analyzer_path):
    trial_num, stim_time = analyzer_pg(analyzer_path)

    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    columns = []
    if type(analyzer.L.param[0]) != str:
        for i in analyzer.L.param:
                columns.append(i[0])
    else:
        columns.append(analyzer.L.param[0])

    trial_info = pd.DataFrame(trial_num, columns=columns)
    return trial_info, stim_time


# def analyzer_pg(analyzer_path):
#     analyzer = load_analyzer(analyzer_path)
#
#     b_flag = 0
#     if analyzer.loops.conds[len(analyzer.loops.conds) - 1].symbol[0] == 'blank':
#         b_flag = 1
#
#     if b_flag == 0:
#         trial_num = np.zeros((len(analyzer.loops.conds) * len(analyzer.loops.conds[0].repeats),
#                              (len(analyzer.loops.conds[0].symbol))))
#     else:
#         trial_num = np.zeros(((len(analyzer.loops.conds) - b_flag) * len(analyzer.loops.conds[0].repeats) +
#                               len(analyzer.loops.conds[-1].repeats),
#                               (len(analyzer.loops.conds[0].symbol))))
#
#     for count in range(0, len(analyzer.loops.conds) - b_flag):
#         trial_vals = np.zeros(len(analyzer.loops.conds[count].symbol))
#
#         for count2 in range(0, len(analyzer.loops.conds[count].symbol)):
#             trial_vals[count2] = analyzer.loops.conds[count].val[count2]
#
#         for count3 in range(0, len(analyzer.loops.conds[count].repeats)):
#             aux_trial = analyzer.loops.conds[count].repeats[count3].trialno
#             trial_num[aux_trial - 1, :] = trial_vals
#
#     for blank_trial in range(0, len(analyzer.loops.conds[-1].repeats)):
#         aux_trial = analyzer.loops.conds[-1].repeats[blank_trial].trialno
#         trial_num[aux_trial - 1, :] = np.array([256, 256])
#
#     stim_time = np.zeros(3)
#     for count4 in range(0, 3):
#         stim_time[count4] = analyzer.P.param[count4][2]
#
#     return trial_num, stim_time


def analyzer_params(analyzer_path):
    analyzer = load_analyzer(analyzer_path)
    params = {}
    for param in analyzer.P.param:
        params[param[0]] = param[2]

    return params


def analyzer_pg_conds(analyzer_path):
    trial_num, stim_time = analyzer_pg(analyzer_path)

    analyzer_complete = sio.loadmat(analyzer_path, squeeze_me=True, struct_as_record=False)
    analyzer = analyzer_complete['Analyzer']

    columns = []
    for i in analyzer.L.param:
        columns.append(i[0])

    trial_info = pd.DataFrame(trial_num, columns=columns)
    return trial_info, stim_time


def gaus(x, a, x0, sigma):
    from scipy import asarray as ar, exp
    return a*exp(-(x-x0)**2/(2*sigma**2))
