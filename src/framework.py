#!/usr/bin/env python

import scipy.io as sio
from model import NaiveBayes

subject = sio.loadmat('data-starplus-04799-v7.mat')

naive_bayes = NaiveBayes(subject)

naive_bayes.train()

#subject2 = sio.loadmat('data-starplus-04820-v7.mat')

naive_bayes.classify()