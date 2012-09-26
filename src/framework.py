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
trial_index = 3
for scan_index in range(54):
	scan = class_subject['data'][trial_index][0][scan_index]
	naive_bayes.classify(scan, scan_index)