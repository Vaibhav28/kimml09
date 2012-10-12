'''The settings module contains application specific configurations.'''

FILE_NAMES = {
    'Original': '../../matlab/data%d-original.mat',
    'NormROIS': '../../matlab/data%d-norm-roi.mat',
    'NormAvergageROIS': '../../matlab/data%d_avergage_rois_norm',
    'AverageROIS': '../../matlab/data%d_avergage_rois',
}

NUM_OF_SUBJECTS = 6

ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']

FIRST_STIMULUS_SCAN_INDEXES = range(10, 20)

SECOND_STIMULUS_SCAN_INDEXES = range(27, 37)

TOTAL_SCAN_INDEXES = FIRST_STIMULUS_SCAN_INDEXES + SECOND_STIMULUS_SCAN_INDEXES