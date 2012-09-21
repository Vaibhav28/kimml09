#!/usr/bin/env python

import scipy.io as sio

'''This is the structure of the data corresponding to every subject.
t = Trial
s = Scan
v = Voxel
subject['data'][t][0][s][v]

For example if we want to access the value of the 1567th voxel of the 20th trial and the 11th scan then we would have the following:

subject['data'][19][0][10][1566]'''

NUM_OF_TRIALS = 54
NUM_OF_VOXELS = 4949
LOW_SCAN_INDEX = 4
HIGH_SCAN_INDEX = 14
ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']

def get_valid_trials(subject):
    '''Returns the valid indexes of the trials for a subject.'''
    return (index for index in range(NUM_OF_TRIALS)
                if subject['info'][0]['cond'][index] > 1)

def get_valid_voxels(subject):
    '''Returns the valid indexes of the voxels for a subject.'''
    return (index for index in range(NUM_OF_VOXELS)
                if subject['meta']['colToROI'][0][0][index] in ROIS)

def run():
    subject = sio.loadmat('data-starplus-04799-v7.mat')
    values = []
    for ti in get_valid_trials(subject):
        for si in range(LOW_SCAN_INDEX, HIGH_SCAN_INDEX):
            for vi in get_valid_voxels(subject):
                values.append(subject['data'][ti][0][si][vi])
    
if __name__ == "__main__":
    run()