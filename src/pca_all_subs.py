from __future__ import division
import scipy.io as sio
from functions import *
from sklearn.naive_bayes import GaussianNB

# Data
print "Loading subjects..."
subjects = [
    sio.loadmat('../matlab/data-starplus-04799-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04820-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04847-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05675-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05680-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05710-v7.mat')
]
print "Loaded subjects!"

# Configuration
FIRST_STIMULUS_SCAN_INDICES = range(10, 20)
SECOND_STIMULUS_SCAN_INDICES = range(27, 37)
COMPONENTS = 200
ROIS = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']

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
# 1: Get list of coordinates
print "Constructing nonempty coordinates across subjects"
coords = getCoordinatesForSubject(subjects[0], ROIS)
validCoords = []
for coord in coords:
    voxels = getVoxelsForCoordinate(coord[0], coord[1], coord[2], subjects)
    if voxels:
        validCoords.append(coord)
trials = getValidTrialIndices(subjects[0])

# 2: Construct Matrix based on those coordinates
scans = []
labels = []
for subject_index, subject in enumerate(subjects):
    voxel_indices = getVoxelsForCoordinates(subject, validCoords)
    print "Appending scans for subject %s" % subject_index
    for trial_index in trials:
        for scan_index in FIRST_STIMULUS_SCAN_INDICES:
            scans.append(getVoxelsForScan(subject, trial_index, scan_index, voxel_indices))
            labels.append(subject['info'][0][trial_index]['firstStimulus'][0])
        for scan_index in SECOND_STIMULUS_SCAN_INDICES:
            scans.append(getVoxelsForScan(subject, trial_index, scan_index, voxel_indices))
            labels.append(u'S') if subject['info'][0][trial_index]['firstStimulus'][0] == 'P' else labels.append(u'P')

# 2b: Normalize / subtract mean?


# 3: PCA
scans_reduced = applyPCA(scans, COMPONENTS)
print "Reduced input data to %s components using PCA" % (len(scans_reduced[0]))
#print "Explained variance is %s" % scans_reduced.explained_variance

# 4: Split last subject...
print "Classification using leave-one-out"
for i in range(len(subjects)):
    print "Leaving out subject %s" % (i + 1)
    scans_training = []
    scans_testing = []
    labels_training = []
    labels_testing = []

    scans_per_subject = (len(scans_reduced) / len(subjects))
    for j in range(len(scans_reduced)):
        training_index_min = i * scans_per_subject
        training_index_max = i * scans_per_subject + scans_per_subject
        if j >= training_index_min and j < training_index_max:
            scans_testing.append(scans_reduced[j])
            labels_testing.append(labels[j])
        else:
            scans_training.append(scans_reduced[j])
            labels_training.append(labels[j])

    print "Split PCA data for training and testing:"
    print len(scans_training)
    print len(scans_testing)

    # 5: Train
    print "Train GaussianNB on training data"
    X = np.array(scans_training)
    Y = np.array(labels_training)
    nb = GaussianNB()
    nb.fit(X, Y)

    # 6: Classify
    print "Classifying using test data"
    correct = 0
    for i in range(len(scans_testing)):
        prediction = nb.predict(scans_testing[i])
        if prediction == labels_testing[i]:
            correct += 1
    print (correct / len(scans_testing)) * 100, "%"
