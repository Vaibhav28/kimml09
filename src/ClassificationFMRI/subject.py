import scipy.io as sio
from settings import ROIS

class Subject:
    """"""

    def __init__(self, matfile):
        """"""
        self.all_data = sio.loadmat(matfile)

    def get_num_of_trials(self):
        """Returns the total number of trials."""
        return self.all_data['meta']['ntrials'][0][0][0][0]

    def get_num_of_voxels(self):
        """Returns the total number of voxels."""
        return self.all_data['meta']['nvoxels'][0][0][0][0]

    def get_valid_trial_indexes(self):
        """Returns the valid indexes of trials for a subject accoring to condition.
        We care only about conditions with value 2 or 3."""
        return [index for index in range(self.get_num_of_trials())
                if self.all_data['info'][0]['cond'][index] > 1]

    def get_first_stimulus(self, trial_index):
        """Returns the first stimulus of a trial with index trial_index. This will
        be either `P' if the subject saw a picture, or `S' if the subject saw a sentence."""
        return self.all_data['info'][0][trial_index]['firstStimulus'][0]

    def get_voxels_of_same_scan(self, trial_index, scan_index):
        """Returns the voxel vector data for the trial with index trial_index
        and scan with index scan_index."""
        return self.all_data['data'][trial_index][0][scan_index]

    def get_voxel_value(self, trial_index, scan_index, voxel_index):
        """Returns a voxel value from the data of the given subject,
        and with trial index, scan_index, and voxel index."""
        return self.all_data['data'][trial_index][0][scan_index][voxel_index]