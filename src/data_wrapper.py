'''
This is the structure of the data corresponding to every subject.
t = Trial
s = Scan
v = Voxel
subject['data'][t][0][s][v]

For example if we want to access the value of the 1567th voxel of the 20th trial
and the 11th scan then we would have the following:

subject['data'][19][0][10][1566]
'''
import pylab as pl
import math

from collections import namedtuple
from sklearn.decomposition import PCA
from array import *


Observation = namedtuple('Observation', 'klass voxels')

FIRST_STIMULUS_SCAN = range(10, 20)
SECOND_STIMULUS_SCAN = range(27, 37)


class DataWrapper:
    ''''''

    def __init__(self, subjects):
        ''''''
        self.subjects = subjects
        self.rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
        self.first_stimulus_index = FIRST_STIMULUS_SCAN
        self.second_stimulus_index = SECOND_STIMULUS_SCAN
        self.features = []
        self.features_p = []
        self.features_s = []

    def get_valid_trial_indexes(self, subject, num_of_trials):
        '''Returns the valid indexes of trials for a subject accoring to condition.
        We care only about conditions with value 2 or 3.'''
        return [index for index in range(num_of_trials)
                if subject['info'][0]['cond'][index] > 1]

    def _get_valid_voxel_indexes(self, num_of_voxels):
        '''Returns the valid indexes of voxels for a subject according to rois.'''
        return [index for index in range(num_of_voxels)
                if self.subject['meta']['colToROI'][0][0][index] in self.rois]

    def _get_first_stimulus(self, subject, trial_index):
        '''Returns the first stimulus of a trial with index trial_index. This will
        be either `P' if the subject saw a picture, or `S' if the subject saw a sentence.'''
        return subject['info'][0][trial_index]['firstStimulus'][0]

    def get_voxels_of_same_scan(self, subject, trial_index, scan_index):
        '''Returns the voxel vector data for the trial with index trial_index
        and scan with index scan_index.'''
        return subject['data'][trial_index][0][scan_index]

    def _extract_features(self, subject, valid_trial_indexes):
        ''''''
        for index, trial_index in enumerate(valid_trial_indexes):
            klass = self._get_first_stimulus(subject, trial_index)
            if klass == 'P':
                for index in self.first_stimulus_index:
                    voxels = self.get_voxels_of_same_scan(subject, trial_index, index)
                    self.features_p.append(voxels)
                    self.features.append(Observation('P', voxels))
                for index in self.second_stimulus_index:
                    voxels = self.get_voxels_of_same_scan(subject, trial_index, index)
                    self.features_s.append(voxels)
                    self.features.append(Observation('S', voxels))
            else:
                for index in self.first_stimulus_index:
                    voxels = self.get_voxels_of_same_scan(subject, trial_index, index)
                    self.features_s.append(voxels)
                    self.features.append(Observation('S', voxels))
                for index in self.second_stimulus_index:
                    voxels = self.get_voxels_of_same_scan(subject, trial_index, index)
                    self.features_p.append(voxels)
                    self.features.append(Observation('P', voxels))
            continue

    def transpose(self, matrix):
        matrix = zip(*matrix)
        return matrix

    def _reduce_features(self):
        '''
        Perform PCA reduction on a set of features. We are only interested
        in the actual data inside the features.

        Returns the reduced set of features.
        '''
        # Transpose for PCA
        # self.features_s = self.transpose(self.features_s)
        # self.features_p = self.transpose(self.features_p)

        # Perform twice, once for P and once for S
        pca = PCA(n_components=250)
        features_s_reduced = pca.fit(self.features_s).transform(self.features_s)
        # print math.fsum(pca.explained_variance_ratio_)
        features_p_reduced = pca.fit(self.features_p).transform(self.features_p)
        # print math.fsum(pca.explained_variance_ratio_)

        # Transpose back for observation grouping
        # features_s_reduced = features_s_reduced.transpose()
        # features_p_reduced = features_p_reduced.transpose()
        print ("Reduced input from [%sX%s] to [%sX%s]" % (len(self.features_s[0]),
                                                          len(self.features_s),
                                                          len(features_s_reduced[0]),
                                                          len(features_s_reduced)))
        # Rebuild features from reduced
        self.features = []
        for voxels_vector in features_s_reduced:
            self.features.append(Observation('S', voxels_vector))
        for voxels_vector in features_p_reduced:
            self.features.append(Observation('P', voxels_vector))

        # Let's draw a 2D plot of the first 2 components
        pl.figure()
        y_map = []
        for i in range(40):
            for j in range(10):
                y_map.append(i)

        for c, k in zip("rg", "ps"):
            target_name = "class %s" % k

            if(k == "p"):
                px = []
                py = []
                for i in range(3):
                    px = px + [point[0] for t, point in enumerate(features_p_reduced) if y_map[t] == i]
                    py = py + [point[1] for t, point in enumerate(features_p_reduced) if y_map[t] == i]
                pl.scatter(px, py, c=c, label=target_name)

            if(k == "s"):
                px = []
                py = []
                for i in range(3):
                    px = px + [point[0] for t, point in enumerate(features_s_reduced) if y_map[t] == i]
                    py = py + [point[1] for t, point in enumerate(features_s_reduced) if y_map[t] == i]
                pl.scatter(px, py, c=c, label=target_name)

        pl.legend()
        pl.title('PCA of "S" trial scans (3 trials, 10 scans per trial)')
        pl.savefig('Fig1.png')

    def extract_values(self):
        ''''''
        for subject in self.subjects:
            num_of_trials = subject['meta']['ntrials'][0][0][0][0]
            valid_trial_indexes = self.get_valid_trial_indexes(subject, num_of_trials)
            self._extract_features(subject, valid_trial_indexes)
            self._reduce_features()  # Should be done per subject

    def get_voxels_of_same_index(self, voxel_index, k):
        '''Returns the voxel values of the same index. For example we want all the voxel
        values with index 0 (means the first voxel value of each voxel vector) in order to
        compute the conditional probability of this voxel: P(V1|Ci).'''
        return [voxels[voxel_index] for klass, voxels in self.features if klass == k]
