#!/usr/bin/env python

from data_wrapper import DataWrapper
from models import NaiveBayes
import scipy.io as sio

# Make sure the *.mat files are in the same directory
subject = sio.loadmat('data-starplus-04799-v7.mat')

data_wrapper = DataWrapper(subject)
data_wrapper.extract_values()

naive_bayes = NaiveBayes(data_wrapper)
naive_bayes.train()

# Classification starts here
if __name__ == '__main__':
    trial_index = 3
    score = 0
    scanset = range(54)  # We probably want only certain scans classified
    for scan_index in scanset:
        scan = subject['data'][trial_index][0][scan_index]
        # Obviously, expected should be derived
        expected_klass = "Sentence"
        predicted_klass = naive_bayes.classify(scan, scan_index)
        if expected_klass == predicted_klass:
            score += 1
    print "Correctly classified %s out of %s scans" % (score, len(scanset))
