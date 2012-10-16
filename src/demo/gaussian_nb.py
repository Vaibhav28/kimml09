#!/usr/bin/env python

from __future__ import division
import scipy.io as sio
import mdp
import numpy as np
from array import array
from sklearn.naive_bayes import GaussianNB

ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
FIRST_STIMULUS_SCAN_INDEXES = range(10, 20)
SECOND_STIMULUS_SCAN_INDEXES = range(22, 32)

def get_number_of_trials(subject):
    ''''''
    return subject['meta']['ntrials']

def get_number_of_voxels(subject):
    ''''''
    return subject['meta']['nvoxels']

def get_valid_trial_indexes(subject, ntrials):
    '''Returns the valid indexes of trials for a subject accoring to condition.
    We care only about conditions with value 2 or 3.'''
    return [index for index in range(ntrials)
            if subject['info'][0]['cond'][index] > 1]

def get_valid_voxel_indexes(subject, nvoxels):
    '''Returns the valid indexes of voxels for a subject according to rois.'''
    return [index for index in range(nvoxels)
            if subject['meta']['colToROI'][0][0][index] in ROIS]

def get_voxels_of_same_scan(subject, trial_index, scan_index):
        '''Returns the voxel vector data for the trial with index trial_index
        and scan with index scan_index.'''
        return subject['data'][trial_index][0][scan_index]

def get_first_stimulus_class(subject, trial_index):
    '''Returns the first stimulus class of a trial with index trial_index. This will
    be either `P' if the subject saw a picture first and then a sentence, or `S' if
    the subject saw a sentence first and then a picture.'''
    return subject['info'][0][trial_index]['firstStimulus'][0]

def extract_features(subject, valid_trial_indexes, valid_voxel_indexes):
    ''''''
    features_p = []
    features_s = []
    for trial_index in valid_trial_indexes:
        klass = get_first_stimulus_class(subject, trial_index)
        if klass == 'P':
            for scan_index in FIRST_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index)
                features_p.append(voxels)
            for scan_index in SECOND_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index)
                features_s.append(voxels)
        else:
            for scan_index in FIRST_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index)
                features_s.append(voxels)
            for scan_index in SECOND_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index)
                features_p.append(voxels)
    return features_p, features_s

def main(subjects, c_subject):
    ''''''
    # TRAINING
    X_P = []
    X_S = []
    nb = GaussianNB()
    for subject in subjects:
        # 1) Feature extraction on training data
        ntrials = get_number_of_trials(subject)
        valid_trial_indexes = get_valid_trial_indexes(subject, ntrials)
        nvoxels = get_number_of_voxels(subject)
        valid_voxel_indexes = get_valid_voxel_indexes(subject, nvoxels)
        features_p, features_s = extract_features(subject, valid_trial_indexes, valid_voxel_indexes)
        features_p = np.array(features_p)
        features_s = np.array(features_s)
        # 2) Fit data to naive bayes classifier
        X = np.vstack((features_p, features_s))
        Y = np.concatenate((np.array([0 for i in range(int(len(X)/2))]), np.array([1 for i in range(int(len(X)/2))])))
        nb.fit(X, Y)

    # CLASSIFICATION
    # 1) Feature extraction on classification data
    c_ntrials = get_number_of_trials(c_subject)
    c_valid_trial_indexes = get_valid_trial_indexes(c_subject, c_ntrials)
    c_nvoxels = get_number_of_voxels(c_subject)
    c_valid_voxel_indexes = get_valid_voxel_indexes(c_subject, c_nvoxels)
    c_features_p, c_features_s = extract_features(c_subject, c_valid_trial_indexes, c_valid_voxel_indexes)
    c_features_p = np.array(c_features_p)
    c_features_s = np.array(c_features_s)
    # 2) Prediction
    X = np.vstack((c_features_p, c_features_s))
    Y = np.concatenate((np.array([0 for i in range(int(len(X)/2))]), np.array([1 for i in range(int(len(X)/2))])))
    correct = 0
    for i in range(len(X)):
        prediction = nb.predict(X[i])
        if prediction == Y[i]:
            correct += 1
    print (correct / len(X)) * 100, "%"

if __name__ == "__main__":
    subjects = [
        sio.loadmat('../../matlab/data1-norm-roi.mat'),
        sio.loadmat('../../matlab/data2-norm-roi.mat'),
        sio.loadmat('../../matlab/data3-norm-roi.mat'),
        sio.loadmat('../../matlab/data4-norm-roi.mat'),
        sio.loadmat('../../matlab/data5-norm-roi.mat'),
    ]
    c_subject = sio.loadmat('../../matlab/data6-norm-roi.mat')
    main(subjects, c_subject)