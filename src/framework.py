#!/usr/bin/env python

import scipy.io as sio

subject = sio.loadmat('data-starplus-04799-v7.mat')

#                         trial scan voxel
# subject_data['data'][0] [t]   [s]  [v]
#print subject['data'][0][0][1][2]

NUMBER_OF_TRIALS = 54
NUMBER_OF_VOXELS = 4949

def get_valid_trials(subject):
    '''Returns the valid indexes of the trials for a subject.'''
    return [i for i in range(NUMBER_OF_TRIALS)
                if subject['info'][0]['cond'][i] > 1 ]

#print get_valid_trials(subject)

ROIs = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']

def get_rois_index(subject):
    '''Returns the valid indexes of the voxels for a subject.'''
    return [i for i in range(NUMBER_OF_VOXELS)
                if subject['meta']['colToROI'][0][0][i] in ROIs]
    
print len(get_rois_index(subject))