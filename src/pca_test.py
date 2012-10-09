#!/usr/bin/env python

from __future__ import division
import scipy.io as sio
import mdp
import numpy as np
from array import array
from sklearn.naive_bayes import GaussianNB

ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
FIRST_STIMULUS_SCAN_INDEXES = range(10, 20)
SECOND_STIMULUS_SCAN_INDEXES = range(27, 37)
COMPONENTS = 200

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

def get_voxels_of_same_scan(subject, trial_index, scan_index, valid_voxel_indexes):
    '''Returns the voxel vector data for the trial with index trial_index
    and scan with index scan_index.'''
    return [subject['data'][trial_index][0][scan_index][voxel_index] 
            for voxel_index in valid_voxel_indexes]

def get_first_stimulus_class(subject, trial_index):
    '''Returns the first stimulus class of a trial with index trial_index. This will
    be either `P' if the subject saw a picture first and then a sentence, or `S' if
    the subject saw a sentence first and then a picture.'''
    return subject['info'][0][trial_index]['firstStimulus'][0]

def extract_features(subject, valid_trial_indexes, valid_voxel_indexes):
    ''''''
    features_p = []
    features_s = []
    for index, trial_index in enumerate(valid_trial_indexes):
        klass = get_first_stimulus_class(subject, trial_index)
        if klass == 'P':
            for scan_index in FIRST_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index, valid_voxel_indexes)
                features_p.append(voxels)
            for scan_index in SECOND_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index, valid_voxel_indexes)
                features_s.append(voxels)
        else:
            for scan_index in FIRST_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index, valid_voxel_indexes)
                features_s.append(voxels)
            for scan_index in SECOND_STIMULUS_SCAN_INDEXES:
                voxels = get_voxels_of_same_scan(subject, trial_index, scan_index, valid_voxel_indexes)
                features_p.append(voxels)
    return features_p, features_s

def apply_pca(features):
    ''''''
    pcanode = mdp.nodes.PCANode(output_dim=COMPONENTS)
    pcanode.train(features)
    pcanode.stop_training()
    return pcanode(features)

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
        # 2) Apply PCA on training data
        pca_out_p = apply_pca(features_p)
        # pca_out_p = [value[:7] for value in pca_out_p]
        # X_P.append(pca_out_p)
        pca_out_s = apply_pca(features_s)
        # pca_out_s = [value[:7] for value in pca_out_s]
        # X_S.append(pca_out_s)
    # X_P = np.vstack((out for out in X_P))
    # X_S = np.vstack((out for out in X_S))
        # 3) Fit data to naive bayes classifier
        X = np.vstack((features_p, features_s))
        Y = np.concatenate((np.array(['P' for i in range(int(len(X)/2))]), np.array(['S' for i in range(int(len(X)/2))])))
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
    # 2) Apply PCA on classification data
    c_pca_out_p = apply_pca(c_features_p)
    # c_pca_out_p = [value[:7] for value in c_pca_out_p]
    c_pca_out_s = apply_pca(c_features_s)
    # c_pca_out_s = [value[:7] for value in c_pca_out_s]
    # 2) Prediction
    X = np.vstack((c_features_p, c_features_s))
    Y = np.concatenate((np.array(['P' for i in range(int(len(X)/2))]), np.array(['S' for i in range(int(len(X)/2))])))
    correct = 0
    for i in range(len(X)):
        prediction = nb.predict(X[i])
        if prediction == Y[i]:
            correct += 1
    print (correct / len(X)) * 100, "%"

if __name__ == "__main__":
    subjects = [
        sio.loadmat('../matlab/data-starplus-04799-v7.mat'),
        sio.loadmat('../matlab/data-starplus-04820-v7.mat'),
        sio.loadmat('../matlab/data-starplus-04847-v7.mat'),
        sio.loadmat('../matlab/data-starplus-05675-v7.mat'),
        sio.loadmat('../matlab/data-starplus-05680-v7.mat'),
        # sio.loadmat('../matlab/data-starplus-05710-v7.mat'),
    ]
    c_subject = sio.loadmat('../matlab/data-starplus-05710-v7.mat')
    main(subjects, c_subject)