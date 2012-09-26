#!/usr/bin/env python
'''
Framework for training and classifying FMRI data.

This framework offers a set of functions to train a classification algorithm
and then use this to classify test data. For more info, please see the article
accompanying this framework.

@since September 2012
'''
from data_wrapper import DataWrapper
from models import NaiveBayes
import scipy.io as sio

subjects = [
    sio.loadmat('../matlab/matlab_avergage_rois_04799.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04820.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04847.mat')
]

data_wrapper = DataWrapper(subjects)
data_wrapper.extract_values()

naive_bayes = NaiveBayes(data_wrapper)
naive_bayes.train()


def get_expected_class(subject, trial_index, scan_index):
    '''
    Checks the subject's info to see what class is expected when
    classifying a scan from a certain trial. Trial index runs from 1 to 54.
    Since we only know 'firstStimulus' for a trial, we'll map all scans before
    [...] to firstStimulus and the rest to the secondStimulus.

    According to data, trial_index 1..27 all have firstStimulus = 'P'.
    Each trial has ~54 scans, so we'll asume that 27 is the halfwayPoint/

    Returns either 'Picture' or 'Sentence'
    '''
    halfwayPoint = 27
    first_stimulus = subject['info'][0][1][trial_index][0]
    if first_stimulus == 'P':
        if scan_index < halfwayPoint:
            return 'Picture'
        else:
            return 'Sentence'
    else:
        if scan_index < halfwayPoint:
            return 'Sentence'
        else:
            return 'Picture'

FIRST_STIMULUS_SCAN = 20
SECOND_STIMULUS_SCAN = 40

# Classification starts here
if __name__ == '__main__':
    class_subject = sio.loadmat('../matlab/matlab_avergage_rois_05675.mat')
    num_of_trials = class_subject['meta']['ntrials'][0][0][0][0]
    valid_trials = data_wrapper.get_valid_trial_indexes(class_subject, num_of_trials)
    score = 0
    for trial_index in valid_trials:
        for scan_index in [FIRST_STIMULUS_SCAN, SECOND_STIMULUS_SCAN]:
            scan = data_wrapper.get_voxels_of_same_scan(class_subject, trial_index, scan_index)
            expected_class = get_expected_class(class_subject, trial_index, scan_index)
            predicted_class = naive_bayes.classify(scan)
            if expected_class == predicted_class:
                score += 1
    print "Correctly classified %s out of %s scans" % (score, len(scanset))
