

class CoordDataWrapper:
    ''''''

    def __init__(self, subjects):
        ''''''
        self.subjects = subjects
        self.features = []

    def extract_subject_coords(self, subject):
        ''''''
        subject_coords = []
        for z in range(8):
            for y in range(64):
                for x in range(64):
                    voxel_index = subject['meta']['coordToCol'][0][0][x][y][z]
                    subject_coords.append(voxel_index)
        return subject_coords

    def extract_values(self):
        ''''''
        # features = []
        # subjects_coords = []
        # for subject_index in range(len(self.subjects)):
        # 	subject_coord = []
        # 	for z in range(8):
        # 		for y in range(64):
        # 			for x in range(64):
        # 				subject_coord.append(self.subjects[subject_index]['meta']['coordToCol'][0][0][x][y][z])
        # 	subjects_coords.append(subject_coord)
        subjects_coords = []
        for subject in self.subjects:
            subjects_coords.append(self.extract_subject_coords(subject))

        voxel_indexes = []
        for coord_index in range(32768):
            something = []
            for subject_index in range(len(self.subjects)):
                something.append(subjects_coords[subject_index][coord_index])
            voxel_indexes.append(something)

        valid_coords = []
        for coord in voxel_indexes:
            if not 0 in coord:
                valid_coords.append(coord)

        # testing against ROI
        for coords in valid_coords:
            for subject_index, coord in enumerate(coords):
                print self.subjects[subject_index]['meta']['colToROI'][0][0][coord]
            print '-----'

        # now we have all the valid voxel indexes for each subject
        # the valid_coords contains lists (605), every list is a coordinate
        # which contains the valid voxel indexes one for each subject
        for i in range(len(self.subjects)):
            internal = []
            for j in range(len(valid_coords)):
                internal.append(valid_coords[j][i])
            self.features.append(internal)
