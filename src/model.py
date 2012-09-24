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

from __future__ import division
from collections import namedtuple
import scipy.io as sio
import scipy.stats as sis
import math

Observation = namedtuple('Observation', 'klass voxels_vector')
ConditionalProbability = namedtuple('ConditionalProbability', 'klass voxel_value')

class NaiveBayes:
	'''This class implements a naive bayes classifier.'''

	def __init__(self, subject):
		self.subject = subject
		self.num_of_trials = None
		self.num_of_voxels = None
		self.valid_trial_indexes = None
		self.valid_voxel_indexes = None
		self.features = []
		self.priori_probabilities = {}
		self.conditional_probabilities = []
		self.rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
		#
		self.first_stimulus_index = 20
		#
		self.second_stimulus_index = 40

	def _get_valid_trials(self):
		'''Returns the valid indexes of trials for a subject accoring to condition.
		We care only about conditions with value 2 or 3.'''
		return [index for index in range(self.num_of_trials)
					if self.subject['info'][0]['cond'][index] > 1]

	def _get_valid_voxels(self):
		'''Returns the valid indexes of voxels for a subject according to rois.'''
		return [index for index in range(self.num_of_voxels)
					if self.subject['meta']['colToROI'][0][0][index] in self.rois]

	def _get_voxel_values(self, trial_indexes, stimulus, voxel):
		'''Returns the absolute value of the same index voxels.'''
		return [math.fabs(self.subject['data'][trial_index][0][stimulus][voxel]) for trial_index in trial_indexes]

	def _get_voxel_vector(self, trial_index, stimulus):
		'''Returns the voxel vector data for the trial with index trial_index
		and scan index as defined by stimulus.'''
		return self.subject['data'][trial_index][0][stimulus]

	def _get_first_stimulus(self, trial_index):
		'''Returns the first stimulus of a trial with index trial_index.'''
		return self.subject['info'][0][trial_index]['firstStimulus'][0]

	def _get_voxels_of_same_index(self, voxel_index, c):
		'''Returns the voxel absolute values of the same index. For example we want all the voxel
		values with index 0 (means the first voxel value of each voxel vector) in order to
		compute the conditional probability of this voxel: P(V1|Ci).'''
		return [voxels[voxel_index] for klass, voxels in self.features if klass == c]

	def _init_values(self):
		'''Extract some basic values that the classifier needs.'''
		self.num_of_trials = self.subject['meta']['ntrials'][0][0][0][0]
		self.num_of_voxels = self.subject['meta']['nvoxels'][0][0][0][0]
		self.valid_trial_indexes = self._get_valid_trials()
		self.valid_voxel_indexes = self._get_valid_voxels()

	def _extract_values(self):
		'''Extract the observations from the subject data. We insert them in a list
		which contains tuples as observations consisting of the class (either P or S)
		and the vector of the voxel values.'''
		for index, trial_index in enumerate(self.valid_trial_indexes):
			klass = self._get_first_stimulus(trial_index)
			if klass == 'P':
				first_observation = Observation('P', self._get_voxel_vector(trial_index, self.first_stimulus_index))
				self.features.append(first_observation)
				second_observation = Observation('S', self._get_voxel_vector(trial_index, self.second_stimulus_index))
				self.features.append(second_observation)
			else:
				first_observation = Observation('S', self._get_voxel_vector(trial_index, self.first_stimulus_index))
				self.features.append(first_observation)
				second_observation = Observation('P', self._get_voxel_vector(trial_index, self.second_stimulus_index))
				self.features.append(second_observation)

	def _compute_priori_probabilities(self):
		'''Computes the priori probabilities of the two classes P and S according to the formula:
		P(Ci) = number of observations in class / number of all observations'''
		total_picture_observations = len([voxels_vector for klass, voxels_vector in self.features if klass == 'P'])
		total_sentence_observations = len([voxels_vector for klass, voxels_vector in self.features if klass == 'S'])
		total_observations = total_picture_observations + total_sentence_observations
		self.priori_probabilities['P'] = total_picture_observations / total_observations
		self.priori_probabilities['S'] = total_sentence_observations / total_observations

	def _compute_conditional_probabilities(self):
		'''Computes the conditional probabilities of each voxel. Only for these voxels that
		we define as valid.'''
		total_sum_picture = 0
		total_sum_sentence = 0
		for index, voxel_index in enumerate(self.valid_voxel_indexes):
			pic_voxels_sum = math.fsum(self._get_voxels_of_same_index(voxel_index=voxel_index, c='P'))
			sen_voxels_sum = math.fsum(self._get_voxels_of_same_index(voxel_index=voxel_index, c='S'))
			total_sum_picture += pic_voxels_sum
			total_sum_sentence += sen_voxels_sum

		pic_cond_probabilities = []
		sen_cond_probabilities = []
		for index, voxel_index in enumerate(self.valid_voxel_indexes):
			pic_voxels_sum = math.fsum(self._get_voxels_of_same_index(voxel_index=voxel_index, c='P'))
			cond_prob = ConditionalProbability('P', pic_voxels_sum / total_sum_picture)
			pic_cond_probabilities.append(cond_prob)
			sen_voxels_sum = math.fsum(self._get_voxels_of_same_index(voxel_index=voxel_index, c='S'))
			cond_prob = ConditionalProbability('S', sen_voxels_sum / total_sum_sentence)
			sen_cond_probabilities.append(cond_prob)
		self.conditional_probabilities.append(pic_cond_probabilities)
		self.conditional_probabilities.append(sen_cond_probabilities)

	def train(self):
		'''Trains the classifier.'''
		self._init_values()
		self._extract_values()
		#self._compute_priori_probabilities()
		#self._compute_conditional_probabilities()

	def classify(self, subject):
		'''Classifies a new scan of data. The following code is just an example as for classification
		training data are used. This needs to be modified.'''
		#####
		#scan = self._get_voxel_vector(trial_index=25, stimulus=34)
		#####
		valid_scan = []
		for index, trial_index in enumerate(self.valid_voxel_indexes):
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
		#
		#
		#
		sum_log_picture = math.fsum(self.distributions['P'])
		sum_log_sentence = math.fsum(self.distributions['S'])
		print "P(Picture|Scan) = ", sum_log_picture
		print "P(Sentence|Scan) = ", sum_log_sentence
		print "Class: ", "Picture" if max(sum_log_picture, sum_log_sentence) == sum_log_picture else "Sentence"

	def practice(self):
		''''''
		self.means_picture = []
		self.means_sentence = []
		for voxel_index in self.valid_voxel_indexes:
			voxel_vector = self._get_voxels_of_same_index(voxel_index, 'P')
			self.means_picture.append(math.fsum(voxel_vector) / len(voxel_vector))
			voxel_vector = self._get_voxels_of_same_index(voxel_index, 'S')
			self.means_sentence.append(math.fsum(voxel_vector) / len(voxel_vector))
		self.standard_deviations_picture = []
		self.standard_deviations_sentence = []
		for index, voxel_index in enumerate(self.valid_voxel_indexes):
			voxel_vector = self._get_voxels_of_same_index(voxel_index, 'P')
			variance = 0
			for voxel_value in voxel_vector:
				variance += ((voxel_value - self.means_picture[index])**2) / (len(voxel_vector)-1)
			self.standard_deviations_picture.append(math.sqrt(variance))
		for index, voxel_index in enumerate(self.valid_voxel_indexes):
			voxel_vector = self._get_voxels_of_same_index(voxel_index, 'S')
			variance = 0
			for voxel_value in voxel_vector:
				variance += ((voxel_value - self.means_sentence[index])**2) / (len(voxel_vector)-1)
			self.standard_deviations_sentence.append(math.sqrt(variance))

if __name__ == "__main__":
	subject = sio.loadmat('data-starplus-04799-v7.mat')
	naive_bayes = NaiveBayes(subject)
	naive_bayes.train()
	naive_bayes.practice()
	subject = sio.loadmat('data-starplus-04820-v7.mat')

	naive_bayes.classify()