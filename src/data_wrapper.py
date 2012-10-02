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

FIRST_STIMULUS_SCAN = 17
SECOND_STIMULUS_SCAN = 34

class DataWrapper:
    ''''''

    def __init__(self, subjects):
        ''''''
        self.subjects = subjects
        self.rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
        self.first_stimulus_index = FIRST_STIMULUS_SCAN
        self.second_stimulus_index = SECOND_STIMULUS_SCAN
        self.features = []

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
                self.features.append(Observation('P', self.get_voxels_of_same_scan(subject,
                                                                                    trial_index,
                                                                                    self.first_stimulus_index)))
                self.features.append(Observation('S', self.get_voxels_of_same_scan(subject,
                                                                                    trial_index,
                                                                                    self.second_stimulus_index)))
            else:
                self.features.append(Observation('S', self.get_voxels_of_same_scan(subject,
                                                                                    trial_index,
                                                                                    self.first_stimulus_index)))
                self.features.append(Observation('P', self.get_voxels_of_same_scan(subject,
                                                                                    trial_index,
                                                                                    self.second_stimulus_index)))

    def extract_values(self):
        ''''''
        for subject in self.subjects:
            num_of_trials = subject['meta']['ntrials'][0][0][0][0]
            valid_trial_indexes = self.get_valid_trial_indexes(subject, num_of_trials)
            self._extract_features(subject, valid_trial_indexes)

    def get_voxels_of_same_index(self, voxel_index, k):
        '''Returns the voxel values of the same index. For example we want all the voxel
        values with index 0 (means the first voxel value of each voxel vector) in order to
        compute the conditional probability of this voxel: P(V1|Ci).'''
        return [voxels[voxel_index] for klass, voxels in self.features if klass == k]
