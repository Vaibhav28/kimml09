#!/usr/bin/env python

from data_wrapper import DataWrapper
from models import NaiveBayes
import scipy.io as sio

subject = sio.loadmat('data-starplus-04799-v7.mat')

data_wrapper = DataWrapper(subject)

data_wrapper.extract_values()

naive_bayes = NaiveBayes(data_wrapper)

naive_bayes.train()