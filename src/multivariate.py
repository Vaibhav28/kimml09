#!/usr/bin/env python
# C:\Users\Robin\Documents\Machine Learning\stuff\src
import scipy.io as sio
import numpy as np
import math as m

# load the matlab files
# file = [
#   'C:\Users\Robin\Documents\Machine Learning\data1-norm-roi'
#   ,'C:\Users\Robin\Documents\Machine Learning\data2-norm-roi'
#   ,'C:\Users\Robin\Documents\Machine Learning\data3-norm-roi'
#   ,'C:\Users\Robin\Documents\Machine Learning\data4-norm-roi'
#   ,'C:\Users\Robin\Documents\Machine Learning\data5-norm-roi'
#   ,'C:\Users\Robin\Documents\Machine Learning\data6-norm-roi'
# ]

# data = [
#       sio.loadmat('../matlab/data1-avg-roi-norm'),
#       sio.loadmat('../matlab/data2-avg-roi-norm'),
#       sio.loadmat('../matlab/data3-avg-roi-norm'),
#       sio.loadmat('../matlab/data4-avg-roi-norm'),
#       sio.loadmat('../matlab/data5-avg-roi-norm'),
#       sio.loadmat('../matlab/data6-avg-roi-norm'),
# ]
data = [
        sio.loadmat('../matlab/data1-selectroi-norm'),
        sio.loadmat('../matlab/data2-selectroi-norm'),
        sio.loadmat('../matlab/data3-selectroi-norm'),
        sio.loadmat('../matlab/data4-selectroi-norm'),
        sio.loadmat('../matlab/data5-selectroi-norm'),
        sio.loadmat('../matlab/data6-selectroi-norm'),
]


#place in scipy format
#data = range(0,6)
#for n in range(0,6):
#   data[n] = sio.loadmat(file[n])

# make variables
correcttotal1 = 0
correcttotal2 = 0
correcttotal3 = 0
correcttotal4 = 0

#repeat the process for 6 subjects
for iteration in range(0, 6):
    # arrr = data array for n-1 subjects
    arrr = range(0, 20)
    # excl = data for the excluded (test) subject
    excl = range(0, 20)
    for n in range(0, 20):
        arrr[n] = [], []
        excl[n] = [], []

    #determine the scans
    firstRange = range(10, 20)
    secondRange = range(22, 32)
    scans = firstRange + secondRange

    #import data into arrr and excl
    exclude = iteration
    for subject in range(len(data)):
        counter = -1
        for scan in scans:
            counter = counter + 1
            for trial in range(0, 53):
                if data[subject]['info'][0, trial]['cond'][0] > 1:
                    if data[subject]['info'][0, trial]['firstStimulus'][0] == 'P':
                        if not subject == exclude:
                            arrr[counter][0].append(data[subject]['data'][trial][0][scan])
                        else:
                            excl[counter][0].append(data[subject]['data'][trial][0][scan])
                    else:
                        if not subject == exclude:
                            arrr[counter][1].append(data[subject]['data'][trial][0][scan])
                        else:
                            excl[counter][1].append(data[subject]['data'][trial][0][scan])

    #set %correct per subject to 0
    correct_per_subject1 = 0
    correct_per_subject2 = 0
    correct_per_subject3 = 0
    correct_per_subject4 = 0

    # test phase
    for ps in range(0, 2):
        for trial in range(0, 20):
            logprob_perscan_pic = [0] * 20
            logprob_perscan_sent = [0] * 20
            for scan in range(0, 20):
                # calculate covariance matrix

                    if scan < 10:
                        #for picture. mat_cov = covariance matrix, mat_det = determinant, mat_inv = inverse matrix, mat_x vector of test input, mat_xminmu = (x-mu)
                        mat_mean = np.subtract(np.matrix(arrr[scan][0]), np.mean(np.matrix(arrr[scan][0]), axis=0))
                        mat_cov = np.dot(np.transpose(mat_mean), mat_mean) / len((arrr[scan][0]))
                        mat_det = np.linalg.det(mat_cov)
                        mat_inv = mat_cov.I
                        mat_x = np.matrix(excl[scan][ps])[trial, 0:25]
                        mat_xminmu = np.matrix(excl[scan][ps])[trial, 0:25] - np.mean(np.matrix(arrr[scan][0]), axis=0)
                        #multivariate equation
                        logprob_perscan_pic[scan] = m.log(2 * m.pi) - 0.5 * m.log(mat_det) - 0.5 * mat_xminmu * mat_inv * np.transpose(mat_xminmu)
                        #for sentence
                        mat_mean = np.subtract(np.matrix(arrr[scan][1]), np.mean(np.matrix(arrr[scan][1]), axis=0))
                        mat_cov = np.dot(np.transpose(mat_mean), mat_mean) / len((arrr[scan][1]))
                        mat_det = np.linalg.det(mat_cov)
                        mat_inv = mat_cov.I
                        mat_x = np.matrix(excl[scan][ps])[trial, 0:25]
                        mat_xminmu = np.matrix(excl[scan][ps])[trial, 0:25] - np.mean(np.matrix(arrr[scan][1]), axis=0)
                        logprob_perscan_sent[scan] = m.log(2 * m.pi) - 0.5 * m.log(mat_det) - 0.5 * mat_xminmu * mat_inv * np.transpose(mat_xminmu)
                    else:
                        #for picture
                        mat_mean = np.subtract(np.matrix(arrr[scan][1]), np.mean(np.matrix(arrr[scan][1]), axis=0))
                        mat_cov = np.dot(np.transpose(mat_mean), mat_mean) / len((arrr[scan][1]))
                        mat_det = np.linalg.det(mat_cov)
                        mat_inv = mat_cov.I
                        mat_x = np.matrix(excl[scan][ps])[trial, 0:25]
                        mat_xminmu = np.matrix(excl[scan][ps])[trial, 0:25] - np.mean(np.matrix(arrr[scan][1]), axis=0)
                        logprob_perscan_pic[scan] = m.log(2 * m.pi) - 0.5 * m.log(mat_det) - 0.5 * mat_xminmu * mat_inv * np.transpose(mat_xminmu)
                        #for sentence
                        mat_mean = np.subtract(np.matrix(arrr[scan][0]), np.mean(np.matrix(arrr[scan][0]), axis=0))
                        mat_cov = np.dot(np.transpose(mat_mean), mat_mean) / len((arrr[scan][0]))
                        mat_det = np.linalg.det(mat_cov)
                        mat_inv = mat_cov.I
                        mat_x = np.matrix(excl[scan][ps])[trial, 0:25]
                        mat_xminmu = np.matrix(excl[scan][ps])[trial, 0:25] - np.mean(np.matrix(arrr[scan][0]), axis=0)
                        logprob_perscan_sent[scan] = m.log(2 * m.pi) - 0.5 * m.log(mat_det) - 0.5 * mat_xminmu * mat_inv * np.transpose(mat_xminmu)

            print ps, trial

            print 'picture % (1-10) = ', np.sum(logprob_perscan_pic[0:10])
            print 'sentence % (1-10) = ', np.sum(logprob_perscan_sent[0:10])
            print 'picture % (10-20) = ', np.sum(logprob_perscan_pic[10:20])
            print 'sentence % (10-20) = ', np.sum(logprob_perscan_sent[10:20])
            # add to correct if correct
            if (np.sum(logprob_perscan_pic[0:10]) > np.sum(logprob_perscan_sent[0:10]) and ps == 0):
                correct_per_subject1 = correct_per_subject1 + 1
            if (np.sum(logprob_perscan_pic[10:20]) > np.sum(logprob_perscan_sent[10:20])and ps == 1):
                correct_per_subject2 = correct_per_subject2 + 1
            if (np.sum(logprob_perscan_pic[10:20]) < np.sum(logprob_perscan_sent[10:20]) and ps == 0):
                correct_per_subject3 = correct_per_subject3 + 1
            if (np.sum(logprob_perscan_pic[0:10]) < np.sum(logprob_perscan_sent[0:10])and ps == 1):
                correct_per_subject4 = correct_per_subject4 + 1
    # add to total
    correcttotal1 = correcttotal1 + correct_per_subject1
    correcttotal2 = correcttotal2 + correct_per_subject2
    correcttotal3 = correcttotal3 + correct_per_subject3
    correcttotal4 = correcttotal4 + correct_per_subject4


# print accuracy
print 'picture % (1-10) = ', correcttotal1 / (6.0 * 20), 'sentence % (1-10) = ', correcttotal4 / (6.0 * 20), 'picture % (10-20) = ', correcttotal2 / (6.0 * 20), 'sentence % (10-20) = ', correcttotal3 / (6.0 * 20)
print 'Total accuracy: ', (correcttotal1 + correcttotal2 + correcttotal3 + correcttotal4) / (6.0 * 80)
