#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import math as m
import scipy.stats as sis
from settings import FILES

# load the matlab files
for datafile in FILES:
    data = [sio.loadmat(FILES[datafile] % (index + 1)) for index in range(6)]

    # make variables
    correcttotal1 = 0
    correcttotal2 = 0
    correcttotal3 = 0
    correcttotal4 = 0
    for iteration in range(0,6):
        arrr = range(0,20)
        excl = range(0,20)
        means = range(0,20) 
        sds = range(0,20)
        for n in range(0,20):
            arrr[n] = [],[]
            excl[n] = [],[]
            means[n] = [],[]
            sds[n] = [],[]
        means = arrr


        scans = range(10,20) + range(22,32)

        #import data
        exclude = iteration
        for subject in range(len(data)):
            counter=-1
            for scan in scans:
                counter = counter + 1
                for trial in range(0,53):
                    if data[subject]['info'][0,trial]['cond'][0]>1:
                        if data[subject]['info'][0,trial]['firstStimulus'][0]=='P': 
                            if not subject == exclude:
                                arrr[counter][0].append(data[subject]['data'][trial][0][scan])
                            else:
                                excl[counter][0].append(data[subject]['data'][trial][0][scan])
                        else:
                            if not subject == exclude:
                                arrr[counter][1].append(data[subject]['data'][trial][0][scan])
                            else:
                                excl[counter][1].append(data[subject]['data'][trial][0][scan])
                    # arrr[counter][0]=np.matrix(arrr[counter][0])
                    # arrr[counter][1]=np.matrix(arrr[counter][1])
            
        #probabilities
        correct_per_subject1 = 0
        correct_per_subject2 = 0
        correct_per_subject3 = 0
        correct_per_subject4 = 0
        for ps in range(0,2):
            for trial in range(0,20):
                logprob_perscan_pic = [0]*20
                logprob_perscan_sent = [0]*20
                for scan in range(0,20):
                    # calculate covariance matrix
                    #covmatr=np.matrix(arrr[scan][0])-
                    for roi in range(0,len(arrr[0][0][0])):
                        if ps == 0:
                            if scans < 10:
                                logprob_perscan_pic[scan]=logprob_perscan_pic[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][0])[trial,roi],np.mean(np.matrix(arrr[scan][0]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][0]),axis=0)[0,roi]))
                                logprob_perscan_sent[scan]=logprob_perscan_sent[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][0])[trial,roi],np.mean(np.matrix(arrr[scan][1]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][1]),axis=0)[0,roi]))
                            else:
                                logprob_perscan_pic[scan]=logprob_perscan_pic[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][0])[trial,roi],np.mean(np.matrix(arrr[scan][1]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][1]),axis=0)[0,roi]))
                                logprob_perscan_sent[scan]=logprob_perscan_sent[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][0])[trial,roi],np.mean(np.matrix(arrr[scan][0]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][0]),axis=0)[0,roi]))
                        else:
                            if scans < 10:
                                logprob_perscan_pic[scan]=logprob_perscan_pic[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][1])[trial,roi],np.mean(np.matrix(arrr[scan][0]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][0]),axis=0)[0,roi]))
                                logprob_perscan_sent[scan]=logprob_perscan_sent[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][1])[trial,roi],np.mean(np.matrix(arrr[scan][1]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][1]),axis=0)[0,roi]))
                            else:
                                logprob_perscan_pic[scan]=logprob_perscan_pic[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][1])[trial,roi],np.mean(np.matrix(arrr[scan][1]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][1]),axis=0)[0,roi]))
                                logprob_perscan_sent[scan]=logprob_perscan_sent[scan]+m.log(sis.norm.pdf(np.matrix(excl[scan][1])[trial,roi],np.mean(np.matrix(arrr[scan][0]),axis=0)[0,roi],np.std(np.matrix(arrr[scan][0]),axis=0)[0,roi]))
                    
                if (np.sum(logprob_perscan_pic[0:10])<np.sum(logprob_perscan_sent[0:10]) and ps ==0): 
                    correct_per_subject1 = correct_per_subject1 + 1
                if (np.sum(logprob_perscan_pic[10:20])>np.sum(logprob_perscan_sent[10:20])and ps==1):   
                    correct_per_subject2 = correct_per_subject2 + 1
                if (np.sum(logprob_perscan_pic[10:20])<np.sum(logprob_perscan_sent[10:20]) and ps==0):
                    correct_per_subject3 = correct_per_subject3 + 1
                if (np.sum(logprob_perscan_pic[0:10])>np.sum(logprob_perscan_sent[0:10])and ps==1):
                    correct_per_subject4 = correct_per_subject4 + 1 
            
        correcttotal1 = correcttotal1 + correct_per_subject1
        
        correcttotal2 = correcttotal2 + correct_per_subject2
        
        correcttotal3 = correcttotal3 + correct_per_subject3
        
        correcttotal4 = correcttotal4 + correct_per_subject4

    # print accuracy
    print "ACCURACY FOR DATAFILE %s" % datafile
    total_picture = (((correcttotal1 / (6.0 * 20)) + (correcttotal2 / (6.0 * 20))) / 2) * 100
    total_sentence = (((correcttotal3 / (6.0 * 20)) + (correcttotal4 / (6.0 * 20))) / 2) * 100
    print 'Picture: %s %%' % total_picture
    print 'Sentence: %s %%' % total_sentence 
    print 'Total accuracy: %s %%' % (((correcttotal1 + correcttotal2 + correcttotal3 + correcttotal4) / (6.0 * 80)) * 100)
    print 