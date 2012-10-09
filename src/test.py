import scipy.io as sio

#rois = []  # Leave empty for all voxels
rois = ['CALC', 'LIPL', 'LT', 'LTRIA', 'LOPER', 'LIPS', 'LDLPFC']
subjects = [
    sio.loadmat('../matlab/data-starplus-04799-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04820-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04847-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05675-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05680-v7.mat'),
    # sio.loadmat('../matlab/data-starplus-05710-v7.mat')
]


def getVoxelsForCoordinate(x, y, z, subjects):
    '''
    Return a list of voxel values for a single coordinate iff all subjects
    have a voxel value for that coordinate.
    Otherwise returns empty list.
    '''
    ret = []
    for subject in subjects:
        voxel_index = subject['meta']['coordToCol'][0][0][x][y][z]
        if voxel_index == 0:
            return []
        ret.append(voxel_index)

    # Made it here? Then none are empty!
    return ret


def getCoordinatesForSubject(subject, rois):
    '''
    Returns a list of non-empty voxel coordinates for a single subject.
    If rois is non-empty, filter the result on regions of interest.
    '''
    coords = []
    for z in range(8):
        for y in range(64):
            for x in range(64):
                voxel_index = subject['meta']['coordToCol'][0][0][x][y][z]
                if voxel_index > 0:
                    voxel_roi = subject['meta']['colToROI'][0][0][voxel_index - 1]
                    if len(rois) == 0 or voxel_roi in rois:
                        coords.append([x, y, z])
    return coords

coords = getCoordinatesForSubject(subjects[0], rois)
validCoords = []
for coord in coords:
    voxels = getVoxelsForCoordinate(coord[0], coord[1], coord[2], subjects)
    if voxels:
        validCoords.append(coord)

# Just some debugging output
# for coord in validCoords:
#     print "Regions for coord (%s,%s,%s)" % (coord[0], coord[1], coord[2])
#     for subject in subjects:
#         voxel_index = subject['meta']['coordToCol'][0][0][coord[0]][coord[1]][coord[2]]
#         print subject['meta']['colToROI'][0][0][voxel_index - 1]

print len(coords)
print len(validCoords)


'''
PCA ALGORITHM
1) Apply filtering on voxels (for each subject)
2) Construct data matrix for PCA:
- each row is a scan from the range of applicable scans (split for type of trial - PS, SP, PS+SP?)
- each column is a value for a certain coordinate
3) Split matrix based on subject
4) Use one part for training
5) Use the other part for testing
'''
