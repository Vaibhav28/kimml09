#!/usr/bin/env python

import scipy.io as sio
from features.extractor import Extractor
from classifiers.naive_bayes import NaiveBayes

class Framework:
	''''''
	file_names = {
		'Original': '../../matlab/data%d-original.mat',
		'NormROIS': '../../matlab/data%d-norm-roi.mat',
		'NormAvergageROIS': '../../matlab/data%d_avergage_rois_norm',
		'AverageROIS': '../../matlab/data%d_avergage_rois',
	}
	num_files = 6

	def __init__(self, classifier, extractor):
		''''''
		self.classifier = classifier
		self.extractor = extractor

	def _load_files(self):
		''''''
		return [sio.loadmat(self.file_names['NormROIS'] % (file_index + 1)) 
					for file_index in range(self.num_files)]

	def execute(self):
		''''''
		# 1) Load the data files
		subject_data = self._load_files()
		# 2) Extract the features
		feature_data = self.extractor.extract_features(subject_data)
		# 3) Train the classifier
		self.classifier.train(feature_data)
		# 4) Classify some data
		self.classifier.classify()	

if __name__ == "__main__":
	# Create a classifier and execute the framework
	nb = NaiveBayes()
	extractor = Extractor()
	framework = Framework(nb, extractor)
	framework.execute()