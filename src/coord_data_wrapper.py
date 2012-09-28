

class CoordDataWrapper:
	''''''

	def __init__(self, subjects):
		''''''
		self.subjects = subjects
		self.features = []

	def extract_values(self):
		''''''
		features = []
		subjects_coords = []
		for subject_index in range(len(self.subjects)):
			subject_coord = []
			for z in range(8):
				for x in range(64):
					for y in range(64):
						subject_coord.append(self.subjects[subject_index]['meta']['coordToCol'][0][0][x][y][z])
			subjects_coords.append(subject_coord)

		voxel_indexes = []
		for coord_index in range(32768):
			something = []
			for subject_index in range(6):
				something.append(subjects_coords[subject_index][coord_index])
			voxel_indexes.append(something)

		valid_coords = []
		for coord in voxel_indexes:
			if not 0 in coord:
				valid_coords.append(coord)

		for valid_coords_list in valid_coords:
			print valid_coords_list