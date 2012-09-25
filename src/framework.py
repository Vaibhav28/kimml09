#!/usr/bin/env python

from data_wrapper import DataWrapper
from models import NaiveBayes
import scipy.io as sio

# Loadmat should use $PATH, but somehow it fails for me (Mattijs)
#subject = sio.loadmat('../../fmri_project/data-starplus-04799-v7.mat')
subject = sio.loadmat('data-starplus-04799-v7.mat')

data_wrapper = DataWrapper(subject)
data_wrapper.extract_values()

naive_bayes = NaiveBayes(data_wrapper)
naive_bayes.train()

if __name__ == '__main__':
    # Classification starts here
    trial_index = 3
    for scan_index in range(54):
        scan = subject['data'][trial_index][0][scan_index]
        naive_bayes.classify(scan, scan_index)
