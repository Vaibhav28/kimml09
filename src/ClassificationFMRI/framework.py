#!/usr/bin/env python

import scipy.io as sio
from subject import Subject
from extractor import Extractor
from classifiers.multivariate import Multivariate
from settings import NUM_OF_SUBJECTS
from settings import FILE_NAMES

class Framework:
    """"""

    def __init__(self, classifier):
        """"""
        self.classifier = classifier
        self.extractor = Extractor()

    def _create_subjects(self):
        """"""
        return [Subject(FILE_NAMES['NormROIS'] % (file_index + 1)) 
                    for file_index in range(NUM_OF_SUBJECTS)]

    def _train(self, classifier, features):
        """"""
        classifier.train(features)

    def _classify(self, classifier):
        """"""
        classifier.classify()

    def execute(self):
        """"""
        # 1) Load the data files
        subjects = self._create_subjects()
        # 2) Extract the features
        self.extractor.extract_features(subjects)
        print len(self.extractor.features['P']), exit()
        # 3) Train the classifier
        self._train(self.classifier, self.extractor.features)
        # 4) Classify some data
        self._classify(self.classifier)

if __name__ == "__main__":
    # Create a classifier, a features extractor, and execute the framework
    classifier = Multivariate()
    framework = Framework(classifier)
    framework.execute()