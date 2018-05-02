import os

from pynwb.behavior import Position
from pynwb.ophys import Fluorescence, RoiResponseSeries, ROITable
from pynwb import NWBHDF5IO, NWBFile
from h5py import File
import numpy as np
from datetime import datetime


base_dir = '/Users/bendichter/Desktop/Schnitzer/data/Example Data'

centroid_fname = 'm655_D11_S1_centroids.mat'
pos_path = os.path.join(base_dir, centroid_fname)

with File(pos_path, 'r') as file:
    pos_data = np.array(file['c']).T

#  not working
pos = Position(name='position', source=centroid_fname, data=pos_data,
               starting_time=0.0, rate=np.nan)


nwbfile = NWBFile(session_start_time=datetime(1900, 1, 1), source=base_dir,
                  session_description=' ', identifier='m655_D11_S1')

module_pos = nwbfile.create_processing_module(name='position', source=' ',
                                              description='position')

module_pos.add_container(pos)

fluorescence_path = os.path.join(base_dir, 'm655_D11_S1.hdf5')
with File(fluorescence_path, 'r') as file:
    fluorescence_data = file['Data']['Images'][:]

data_names = ['cellImages', 'cellTraces', 'centroids', 'eventTimes']
mat_data = {}
mat_output_fpath = os.path.join(base_dir, 'm655_D11_S1_output.mat')
with File(mat_output_fpath, 'r') as file:
    for name in data_names:
        mat_data[name] = file['output'][name][:]

roi_table = ROITable()
for i, image in enumerate(mat_data['cellImages']):
    roi_table.add_row(name=str(i), pixel_mask=image, image_mask=image)



#RoiResponseSeries(data=mat_data['cellTraces'], name='all cells',
#                  source=mat_output_fpath, unit='unknown', rois='huh?')

#  not working
nwbfile.add_acquisition(Fluorescence(source=' ', data=fluorescence_data))

fname_out = 'm655_D11_S1.nwb'
with NWBHDF5IO(fname_out, 'w') as io:
    io.write(nwbfile)

