#!/usr/bin/env python
'''
Framework for training and classifying FMRI data.

This framework offers a set of functions to train a classification algorithm
and then use this to classify test data. For more info, please see the article
accompanying this framework.

@since September 2012
'''
from __future__ import division

import scipy.io as sio
from data_wrapper import DataWrapper
from models import NaiveBayes
from functions import get_expected_class
from data_wrapper import FIRST_STIMULUS_SCAN
from data_wrapper import SECOND_STIMULUS_SCAN
import math

files = [
    '../matlab/matlab_avergage_rois_04799.mat',
    '../matlab/matlab_avergage_rois_04820.mat',
    '../matlab/matlab_avergage_rois_04847.mat',
    '../matlab/matlab_avergage_rois_05675.mat',
    '../matlab/matlab_avergage_rois_05680.mat',
    '../matlab/matlab_avergage_rois_05710.mat'
]

'''
For our model, we first train it using a subset of data and then
classify it on some (other) subset of data. Make sure to
split these sets at the top of the following loop.
'''
tscore = 0
tcount = 0
for i in range(len(files)):
    # Select 1 file for testing
    test_data = files[i]
    subjects = []

    # Use the rest for training
    for j in range(len(files)):
        if j != i:
            subjects.append(sio.loadmat(files[j]))

    data_wrapper = DataWrapper(subjects)
    data_wrapper.extract_values()

    naive_bayes = NaiveBayes(data_wrapper)
    naive_bayes.train()
    print "Classifier trained using %s subjects:" % (len(subjects))

    '''
    From here on, we start classification
    '''
    class_subject = sio.loadmat(test_data)
    num_of_trials = class_subject['meta']['ntrials'][0][0][0][0]
    valid_trials = data_wrapper.get_valid_trial_indexes(class_subject, num_of_trials)
    score = 0
    counter = 0
    for trial_index in valid_trials:
        for scan_index_vector in [FIRST_STIMULUS_SCAN, SECOND_STIMULUS_SCAN]:
            sum_sum_log_picture = 0
            sum_sum_log_sentence = 0
            expected_class = get_expected_class(class_subject, trial_index, scan_index_vector[int(math.floor(len(scan_index_vector)/2))])
            for scan_index in scan_index_vector:
                scan = data_wrapper.get_voxels_of_same_scan(class_subject, trial_index, scan_index)
                sum_log_picture, sum_log_sentence = naive_bayes.classify(scan)
                sum_sum_log_picture += sum_log_picture
                sum_sum_log_sentence += sum_log_sentence
            predicted_class = "Picture" if max(sum_sum_log_picture, sum_sum_log_sentence) == sum_sum_log_picture else "Sentence"
            if expected_class == predicted_class:
                score += 1
            counter += 1
    print "Correctly classified %s out of %s scans" % (score, counter)
    print "Score: %s%%" % (score / counter * 100.0)
    tscore += score
    tcount += counter

print "Overall classified %s out of %s scans" % (tscore, tcount)
print "Total score: %s%%" % (tscore / tcount * 100.0)
