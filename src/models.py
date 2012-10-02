from __future__ import division
from collections import namedtuple
import scipy
import scipy.stats as sis
import math

ConditionalProbability = namedtuple('ConditionalProbability', 'klass voxel_value')

NUM_OF_VOXELS = 7

class NaiveBayes:
    '''This class implements a naive bayes classifier.'''

    def __init__(self, data_wrapper):
        ''''''
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
        for voxel_index in range(NUM_OF_VOXELS):
            voxel_vector = self.data_wrapper.get_voxels_of_same_index(  voxel_index, 'P')
            self.means_picture.append(scipy.mean(voxel_vector))
            voxel_vector = self.data_wrapper.get_voxels_of_same_index(voxel_index, 'S')
            self.means_sentence.append(scipy.mean(voxel_vector))

    def _compute_standard_deviations(self):
        ''''''
        for index in range(NUM_OF_VOXELS):
            voxel_vector = self.data_wrapper.get_voxels_of_same_index(index, 'P')
            self.standard_deviations_picture.append(scipy.std(voxel_vector))
        for index in range(NUM_OF_VOXELS):
            voxel_vector = self.data_wrapper.get_voxels_of_same_index(index, 'S')
            self.standard_deviations_sentence.append(scipy.std(voxel_vector))

    def train(self):
        '''Trains the classifier by computing the priori probabilities of the classes,
        the means for each voxel index, and the standard deviations for each vector index.'''
        self._compute_priori_probabilities()
        self._compute_means()
        self._compute_standard_deviations()

    def _classification_result(self, sum_log_picture, sum_log_sentence):
        ''''''
        return "Picture" if max(sum_log_picture, sum_log_sentence) == sum_log_picture else "Sentence"

    def classify(self, scan):
        '''Classifies a new scan of data.'''
        self.distributions = {}
        distribution_picture = []
        distribution_sentence = []
        for index, voxel_value in enumerate(scan):
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
        return (sum_log_picture, sum_log_sentence)
        # klass = self._classification_result(sum_log_picture, sum_log_sentence)
        # print "##########"
        # print "P(Picture|Scan) = ", sum_log_picture
        # print "P(Sentence|Scan) = ", sum_log_sentence
        # print "Class: ", klass
        # return klass
