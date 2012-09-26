#!/usr/bin/env python

from data_wrapper import DataWrapper
from models import NaiveBayes
import scipy.io as sio

files = [
	'../matlab/matlab_avergage_rois_04799.mat',
	# 'matlab_avergage_rois_04820.mat',
	# 'matlab_avergage_rois_04847.mat'
]

for file in files:
	subject = sio.loadmat(file)

	data_wrapper = DataWrapper(subject)
	data_wrapper.extract_values()

	naive_bayes = NaiveBayes(data_wrapper)
	naive_bayes.train()


class_subject = sio.loadmat('matlab_avergage_rois_04799.mat')

# Classification starts here
if __name__ == '__main__':
    trial_index = 3
    score = 0
    scanset = range(54)  # We probably want only certain scans classified
    for scan_index in scanset:
        scan = class_subject['data'][trial_index][0][scan_index]
        # Obviously, expected should be derived
        expected_klass = "Sentence"
        predicted_klass = naive_bayes.classify(scan, scan_index)
        if expected_klass == predicted_klass:
            score += 1
    print "Correctly classified %s out of %s scans" % (score, len(scanset))