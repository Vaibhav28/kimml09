from __future__ import division
import scipy.io as sio
import pylab
from scipy import std as sstd
from scipy import mean as smean
from functions import *
from sklearn.naive_bayes import GaussianNB
#from sklearn.preprocessing import normalize
from math import fsum

# Data
print "Loading subjects..."

subjects = [
    sio.loadmat('demo/matlab/data1-original.mat'),
    sio.loadmat('demo/matlab/data2-original.mat'),
    sio.loadmat('demo/matlab/data3-original.mat'),
    sio.loadmat('demo/matlab/data4-original.mat'),
    sio.loadmat('demo/matlab/data5-original.mat'),
    sio.loadmat('demo/matlab/data6-original.mat')
]
# subjects = [
#     sio.loadmat('../matlab/matlab_avergage_rois_04799.mat'),
#     sio.loadmat('../matlab/matlab_avergage_rois_04820.mat'),
#     sio.loadmat('../matlab/matlab_avergage_rois_04847.mat'),
#     sio.loadmat('../matlab/matlab_avergage_rois_05675.mat'),
#     sio.loadmat('../matlab/matlab_avergage_rois_05680.mat'),
#     sio.loadmat('../matlab/matlab_avergage_rois_05710.mat')
# ]
print "Loaded subjects!"

# Configuration
FIRST_STIMULUS_SCAN_INDICES = range(10, 20)
SECOND_STIMULUS_SCAN_INDICES = range(27, 37)
COMPONENTS = 148

# Rois are ignored if filter coords is false
#ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
ROIS = []

FLAG_PCA = True
FLAG_NORM = False
FLAG_FILTER_COORDS = False
FLAG_FILTER_INDICES = True

# If true, average the prediction per set of scans
FLAG_PER_SCAN = False

# Just some debugging output
# for coord in validCoords:
#     print "Regions for coord (%s,%s,%s)" % (coord[0], coord[1], coord[2])
#     for subject in subjects:
#         voxel_index = subject['meta']['coordToCol'][0][0][coord[0]][coord[1]][coord[2]]
#         print subject['meta']['colToROI'][0][0][voxel_index - 1]

'''
PCA ALGORITHM
1) Apply filtering on voxels (for each subject)
2) Construct data matrix for PCA:
- each row is a scan from the range of applicable scans (split for type of trial - PS, SP, PS+SP?)
- each column is a value for a certain coordinate
3) PCA
4) Split matrix based on subject
5) Use one part for training
6) Use the other part for testing
'''
# 1: Get list of all indices
validCoords = []
trials = getValidTrialIndices(subjects[0])
if FLAG_FILTER_COORDS:
    print "Constructing nonempty coordinates across subjects"
    coords = getCoordinatesForSubject(subjects[0], ROIS)
    print len(coords)
    for coord in coords:
        voxels = getVoxelsForCoordinate(coord[0], coord[1], coord[2], subjects)
        if voxels:
            validCoords.append(coord)
    if len(validCoords) == 0:
        print "No valid coordinates present"
        quit()

if FLAG_FILTER_INDICES:
    '''
    Take all voxels that are present for each subject
    '''
    lengths = []
    for subject in subjects:
        lengths.append(len(subject['meta']['colToROI'][0][0]))
    voxel_indices = range(1, min(lengths))
    print lengths
    # Filter ROIS
    if len(ROIS) > 0:
        roi_indices = []
        for index in voxel_indices:
            if subjects[0]['meta']['colToROI'][0][0][index - 1] in ROIS:
                roi_indices.append(index)
        voxel_indices = roi_indices

print len(voxel_indices)
# 2: Construct Matrix based on those coordinates
scans = []
labels = []
for subject_index, subject in enumerate(subjects):
    subject_scans = []
    if not FLAG_FILTER_COORDS:
        validCoords = getCoordinatesForSubject(subject, [])

    print "Appending scans for subject %s" % subject_index
    for trial_index in trials:
        for scan_index in FIRST_STIMULUS_SCAN_INDICES:
            subject_scans.append(getVoxelsForScan(subject, trial_index, scan_index, voxel_indices))
            labels.append(subject['info'][0][trial_index]['firstStimulus'][0])
        for scan_index in SECOND_STIMULUS_SCAN_INDICES:
            subject_scans.append(getVoxelsForScan(subject, trial_index, scan_index, voxel_indices))
            labels.append(u'S') if subject['info'][0][trial_index]['firstStimulus'][0] == 'P' else labels.append(u'P')

    # 2b: Normalize / subtract mean?
    if FLAG_NORM:
        mean = smean([scan[0] for scan in subject_scans])
        std = sstd([scan[0] for scan in subject_scans])
        for scan_index, scan in enumerate(subject_scans):
            for voxel_index, voxel in enumerate(scan):
                subject_scans[scan_index][voxel_index] = (voxel - mean) / std

    # Add scans to complete set
    scans += subject_scans

# 3: PCA
original_scans = scans
vec_x = []
vec_y = []

for n in range(COMPONENTS, COMPONENTS + 1):
    scans = applyPCA(original_scans, n + 1)
    #print "Reduced input data to %s components using PCA" % (len(scans[0]))
    #print "Explained variance is %s" % scans_reduced.explained_variance

    # 4: Split last subject...
    total_correct = 0
    total_scans = 0
    #print "Classification using leave-one-out"
    for i in range(len(subjects)):
        #print "Leaving out subject %s" % (i + 1)
        scans_training = []
        scans_testing = []
        labels_training = []
        labels_testing = []

        scans_per_subject = (len(scans) / len(subjects))
        for j in range(len(scans)):
            training_index_min = i * scans_per_subject
            training_index_max = i * scans_per_subject + scans_per_subject
            if j >= training_index_min and j < training_index_max:
                scans_testing.append(scans[j])
                labels_testing.append(labels[j])
            else:
                scans_training.append(scans[j])
                labels_training.append(labels[j])

        #print "Split data for training and testing:"
        #print len(scans_training)
        #print len(scans_testing)

        # 5: Train
        #print "Train GaussianNB on training data"
        X = np.array(scans_training)
        Y = np.array(labels_training)
        nb = GaussianNB()
        nb.fit(X, Y)

        if FLAG_PER_SCAN:
            # 6: Classify per scan
            #print "Classifying using test data"
            correct = 0
            for i in range(len(scans_testing)):
                prediction = nb.predict(scans_testing[i])
                if prediction == labels_testing[i]:
                    correct += 1
            #print (correct / len(scans_testing)) * 100, "%"
            total_correct += correct
            total_scans += len(scans_testing)
        else:
            # 6: Classify per group
            #print "Classifying using test data"
            correct = 0
            count = 0
            sum_p = 0
            sum_s = 0
            for i in range(len(scans_testing)):
                prediction = nb.predict_log_proba(scans_testing[i])
                sum_p += prediction[0][0]
                sum_s += prediction[0][1]

                if i % 10 == 9:
                    group_prediction = 'P' if sum_p > sum_s else 'S'
                    sum_p = 0
                    sum_s = 0
                    count += 1
                    if group_prediction == labels_testing[i]:
                        correct += 1

            #print (correct / count) * 100, "%"
            total_correct += correct
            total_scans += count

    vec_x.append(n)
    vec_y.append((total_correct / total_scans) * 100)
    print "%s,%s%%" % (n, (total_correct / total_scans) * 100)


# Plot?
pylab.plot(vec_x, vec_y)
pylab.xlabel('Number of components')
pylab.ylabel('Accuracy (%)')
pylab.title('PCA training results for different amounts of components')
pylab.grid(True)
pylab.savefig('pca_plot')
pylab.show()
