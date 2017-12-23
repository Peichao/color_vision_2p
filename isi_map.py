import os
import glob
import functions

animal = 'AE1'
unit = '000'
exp = '004'

main_dir = 'F:/NHP'
mov_folder = os.path.join(main_dir, animal, 'ISI data', 'u%s_e%s' % (unit, exp))
mov_file = glob.glob(mov_folder + '/trial_response.mat')[0]

analyzer_path = os.path.join(main_dir, animal, 'AnalyzerFiles', animal, '%s_u%s_%s' % (animal, unit, exp) + '.analyzer')
trial_info, stim_time = functions.analyzer_pg_conds(analyzer_path)
