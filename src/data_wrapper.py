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

from collections import namedtuple

Observation = namedtuple('Observation', 'klass voxels')


class DataWrapper:
    ''''''

    def __init__(self, subject):
        ''''''
        self.subject = subject
        self.num_of_voxels = None
        self.num_of_trials = None
        self.valid_trial_indexes = None
        self.valid_voxel_indexes = None
        self.rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
        self.first_stimulus_index = 20
        self.second_stimulus_index = 40
        self.features = []

    def _get_valid_trial_indexes(self):
        '''Returns the valid indexes of trials for a subject accoring to condition.
        We care only about conditions with value 2 or 3.'''
        return [index for index in range(self.num_of_trials)
                if self.subject['info'][0]['cond'][index] > 1]

    def _get_valid_voxel_indexes(self):
        '''Returns the valid indexes of voxels for a subject according to rois.'''
        return [index for index in range(self.num_of_voxels)
                if self.subject['meta']['colToROI'][0][0][index] in self.rois]

    def _get_first_stimulus(self, trial_index):
        '''Returns the first stimulus of a trial with index trial_index. This will
        be either `P' if the subject saw a picture, or `S' if the subject saw a sentence.'''
        return self.subject['info'][0][trial_index]['firstStimulus'][0]

    # non_private for now
    def get_voxels_of_same_scan(self, trial_index, scan_index):
        '''Returns the voxel vector data for the trial with index trial_index
        and scan with index scan_index.'''
        return self.subject['data'][trial_index][0][scan_index]

    def _extract_features(self):
        ''''''
        features = []
        for index, trial_index in enumerate(self.valid_trial_indexes):
            klass = self._get_first_stimulus(trial_index)
            if klass == 'P':
                features.append(Observation('P', self.get_voxels_of_same_scan(trial_index,
                                                                              self.first_stimulus_index)))
                features.append(Observation('S', self.get_voxels_of_same_scan(trial_index,
                                                                              self.second_stimulus_index)))
            else:
                features.append(Observation('S', self.get_voxels_of_same_scan(trial_index,
                                                                              self.first_stimulus_index)))
                features.append(Observation('P', self.get_voxels_of_same_scan(trial_index,
                                                                              self.second_stimulus_index)))
        return features

    def extract_values(self):
        ''''''
        self.num_of_trials = self.subject['meta']['ntrials'][0][0][0][0]
        self.num_of_voxels = self.subject['meta']['nvoxels'][0][0][0][0]
        self.valid_trial_indexes = self._get_valid_trial_indexes()
        self.valid_voxel_indexes = self._get_valid_voxel_indexes()
        self.features = self._extract_features()

    def get_voxels_of_same_index(self, voxel_index, k):
        '''Returns the voxel values of the same index. For example we want all the voxel
        values with index 0 (means the first voxel value of each voxel vector) in order to
        compute the conditional probability of this voxel: P(V1|Ci).'''
        return [voxels[voxel_index] for klass, voxels in self.features if klass == k]
