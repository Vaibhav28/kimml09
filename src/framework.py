#!/usr/bin/env python
'''
Framework for training and classifying FMRI data.

This framework offers a set of functions to train a classification algorithm
and then use this to classify test data. For more info, please see the article
accompanying this framework.

@since September 2012
'''
import scipy.io as sio

from data_wrapper import DataWrapper
from coord_data_wrapper import CoordDataWrapper
from models import NaiveBayes
from functions import get_expected_class
from data_wrapper import FIRST_STIMULUS_SCAN
from data_wrapper import SECOND_STIMULUS_SCAN

# List of matlab data files
subjects = [
    sio.loadmat('../matlab/matlab_avergage_rois_04799.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04820.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04847.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_05675.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_05680.mat'),
    # sio.loadmat('../matlab/matlab_avergage_rois_05710.mat'),
    # sio.loadmat('../matlab/data-starplus-04799-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-04820-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-04847-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-05675-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-05680-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-05710-v7.mat')
]

# coord_data_wrapper = CoordDataWrapper(subjects)
# coord_data_wrapper.extract_values()

data_wrapper = DataWrapper(subjects)
data_wrapper.extract_values()

naive_bayes = NaiveBayes(data_wrapper)
naive_bayes.train()

if __name__ == '__main__':
    '''
    From here on, we start classification
    '''
    class_subject = sio.loadmat('../matlab/matlab_avergage_rois_05710.mat')
    num_of_trials = class_subject['meta']['ntrials'][0][0][0][0]
    valid_trials = data_wrapper.get_valid_trial_indexes(class_subject, num_of_trials)
    score = 0
    counter = 0
    for trial_index in valid_trials:
        for scan_index in [FIRST_STIMULUS_SCAN, SECOND_STIMULUS_SCAN]:
            # print len(data_wrapper.subjects[5]['data'][trial_index][0][scan_index])
            scan = data_wrapper.get_voxels_of_same_scan(class_subject, 5, trial_index, scan_index)
            expected_class = get_expected_class(class_subject, trial_index, scan_index)
            predicted_class = naive_bayes.classify(scan)
            if expected_class == predicted_class:
                score += 1
            counter += 1
    print "Correctly classified %s out of %s scans" % (score, counter)
