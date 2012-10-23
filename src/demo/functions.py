from mdp.nodes import PCANode
import numpy as np


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
    first_stimulus = subject['info'][0][trial_index]['firstStimulus'][0]
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


def getVoxelsForCoordinate(x, y, z, subjects):
    '''
    Return a list of voxel values for a single coordinate iff all subjects
    have a voxel value for that coordinate.
    Otherwise returns empty list.
    '''
    ret = []
    for subject in subjects:
        voxel_index = subject['meta']['coordToCol'][0][0][x][y][z]
        if voxel_index == 0:
            return []
        ret.append(voxel_index)

    # Made it here? Then none are empty!
    return ret


def getCoordinatesForSubject(subject, rois):
    '''
    Returns a list of non-empty voxel coordinates for a single subject.
    If rois is non-empty, filter the result on regions of interest.
    '''
    coords = []
    for z in range(8):
        for y in range(64):
            for x in range(64):
                voxel_index = subject['meta']['coordToCol'][0][0][x][y][z]
                if voxel_index > 0:
                    voxel_roi = subject['meta']['colToROI'][0][0][voxel_index - 1]
                    if len(rois) == 0 or voxel_roi in rois:
                        coords.append([x, y, z])
    return coords


def getVoxelsForScan(subject, trial_index, scan_index, voxel_indices):
    '''
    Returns the voxel vector data for the trial with index trial_index
    and scan with index scan_index.

    Optionally, a list of voxel_indices can be used to further filter the
    result.
    '''
    if voxel_indices:
        voxels = []
        for voxel_index in voxel_indices:
            voxels.append(subject['data'][trial_index][0][scan_index][voxel_index - 1])
        return voxels

    return subject['data'][trial_index][0][scan_index]


def getValidTrialIndices(subject):
    '''
    Returns the valid indices of trials for a subject according to condition.
    We care only about conditions with value 2 or 3.
    '''
    return [index for index in range(subject['meta']['ntrials'])
            if subject['info'][0]['cond'][index] > 1]


def getVoxelsForCoordinates(subject, coords):
    ret = []
    for coord in coords:
        ret.append(subject['meta']['coordToCol'][0][0][coord[0]][coord[1]][coord[2]])
    return ret


def applyPCA(features, components):
    '''
    Dimension reduction of a matrix using pca
    '''
    features = np.array(features)
    pcanode = PCANode(output_dim=components)
    pcanode.train(features)
    pcanode.stop_training()

    print "Explained variance: %s" % pcanode.explained_variance
    return pcanode(features)
