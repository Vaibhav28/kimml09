from coord_data_wrapper import CoordDataWrapper
import scipy.io as sio

subjects = [
    sio.loadmat('../matlab/matlab_avergage_rois_04799.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04820.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_04847.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_05675.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_05680.mat'),
    sio.loadmat('../matlab/matlab_avergage_rois_05710.mat')
]

coord = CoordDataWrapper(subjects)

coord.extract_values()