from coord_data_wrapper import CoordDataWrapper
import scipy.io as sio

subjects = [
    sio.loadmat('../matlab/data-starplus-04799-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04820-v7.mat'),
    sio.loadmat('../matlab/data-starplus-04847-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05675-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05680-v7.mat'),
    sio.loadmat('../matlab/data-starplus-05710-v7.mat')
]

coord = CoordDataWrapper(subjects)

coord.extract_values()
