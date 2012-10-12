from settings import NUM_OF_SUBJECTS
from settings import TOTAL_SCAN_INDEXES
import numpy as np

class Extractor:
    """"""

    def __init__(self):
        """"""
        self.features = {'P': [], 'S': []}

    def extract_features(self, subjects):
        """"""
        excluded = subjects[0]
        for scan_index in TOTAL_SCAN_INDEXES:
            for subject in subjects:
                for trial_index in subject.get_valid_trial_indexes():
                    if subject is not excluded:
                        klass = subject.get_first_stimulus(trial_index)
                        if klass == 'P':
                            self.features['P'].append(subject.get_voxels_of_same_scan(trial_index, scan_index))
                        else:
                            self.features['S'].append(subject.get_voxels_of_same_scan(trial_index, scan_index))