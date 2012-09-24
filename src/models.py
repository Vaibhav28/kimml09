from __future__ import division
from collections import namedtuple
import scipy.stats as sis
import math

ConditionalProbability = namedtuple('ConditionalProbability', 'klass voxel_value')

class NaiveBayes:
	'''This class implements a naive bayes classifier.'''

	def __init__(self, data_wrapper):
		self.data_wrapper = data_wrapper
		self.priori_probabilities = {}
		self.means_picture = []
		self.means_sentence = []
		self.standard_deviations_picture = []
		self.standard_deviations_sentence = []

	def _compute_priori_probabilities(self):
		'''Computes the priori probabilities of the two classes P and S according to the formula:
		P(Ci) = number of observations in class / number of all observations'''
		total_picture_observations = len([voxels_vector for klass, voxels_vector in self.data_wrapper.features if klass == 'P'])
		total_sentence_observations = len([voxels_vector for klass, voxels_vector in self.data_wrapper.features if klass == 'S'])
		total_observations = total_picture_observations + total_sentence_observations
		self.priori_probabilities['P'] = total_picture_observations / total_observations
		self.priori_probabilities['S'] = total_sentence_observations / total_observations

	def _compute_means(self):
		''''''
		for voxel_index in self.data_wrapper.valid_voxel_indexes:
			voxel_vector = self.data_wrapper.get_voxels_of_same_index(voxel_index, 'P')
			self.means_picture.append(math.fsum(voxel_vector) / len(voxel_vector))
			voxel_vector = self.data_wrapper.get_voxels_of_same_index(voxel_index, 'S')
			self.means_sentence.append(math.fsum(voxel_vector) / len(voxel_vector))

	def _compute_standard_deviations(self):
		''''''
		for index, voxel_index in enumerate(self.data_wrapper.valid_voxel_indexes):
			voxel_vector = self.data_wrapper.get_voxels_of_same_index(voxel_index, 'P')
			variance = 0
			for voxel_value in voxel_vector:
				variance += ((voxel_value - self.means_picture[index])**2) / (len(voxel_vector)-1)
			self.standard_deviations_picture.append(math.sqrt(variance))
		for index, voxel_index in enumerate(self.data_wrapper.valid_voxel_indexes):
			voxel_vector = self.data_wrapper.get_voxels_of_same_index(voxel_index, 'S')
			variance = 0
			for voxel_value in voxel_vector:
				variance += ((voxel_value - self.means_sentence[index])**2) / (len(voxel_vector)-1)
			self.standard_deviations_sentence.append(math.sqrt(variance))

	def train(self):
		'''Trains the classifier by computing the priori probabilities of the classes,
		the means for each voxel index, and the standard deviations for each vector index.'''
		self._compute_priori_probabilities()
		self._compute_means()
		self._compute_standard_deviations()

	def classify(self, scan):
		'''Classifies a new scan of data.'''
		valid_scan = []
		for index, trial_index in enumerate(self.data_wrapper.valid_voxel_indexes):
			valid_scan.append(scan[trial_index])
		self.distributions = {}
		distribution_picture = []
		distribution_sentence = []
		for index, voxel_value in enumerate(valid_scan):
			distribution_picture.append(sis.norm.pdf(voxel_value,
													 self.means_picture[index],
													 self.standard_deviations_picture[index]))
			distribution_sentence.append(sis.norm.pdf(voxel_value,
													  self.means_sentence[index],
													  self.standard_deviations_sentence[index]))
		self.distributions['P'] = [math.log(value) for value in distribution_picture]
		self.distributions['S'] = [math.log(value) for value in distribution_sentence]
		sum_log_picture = math.fsum(self.distributions['P'])
		sum_log_sentence = math.fsum(self.distributions['S'])
		print "P(Picture|Scan) = ", sum_log_picture
		print "P(Sentence|Scan) = ", sum_log_sentence
		print "Class: ", "Picture" if max(sum_log_picture, sum_log_sentence) == sum_log_picture else "Sentence"