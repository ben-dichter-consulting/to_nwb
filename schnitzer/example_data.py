import os

from pynwb.behavior import Position, SpatialSeries
from pynwb.ophys import Fluorescence, RoiResponseSeries, OpticalChannel, \
    TwoPhotonSeries, ImageSegmentation
from pynwb.image import ImageSeries
from pynwb import NWBHDF5IO, NWBFile
from h5py import File
import numpy as np
from datetime import datetime


base_dir = '/Users/bendichter/Desktop/Schnitzer/data/Example Data'

nwbfile = NWBFile(session_start_time=datetime(1900, 1, 1), source=base_dir,
                  session_description=' ', identifier='m655_D11_S1')

nwbfile.add_acquisition(ImageSeries(name='video', source='m655_D11_S1.avi',
                                    format='external',
                                    external_file=['m655_D11_S1.avi'],
                                    starting_time=0.0, rate=np.nan,
                                    starting_frame=[0],
                                    dimension=[250, 250]))

centroid_fname = 'm655_D11_S1_centroids.mat'
pos_path = os.path.join(base_dir, centroid_fname)

with File(pos_path, 'r') as file:
    pos_data = np.array(file['c']).T
spatial_series = SpatialSeries(name='position', source='source',
                               data=pos_data, starting_time=0.0, rate=5.0,
                               units='unknown', reference_frame='unknown')

pos = Position(name='position', source=centroid_fname,
               spatial_series=spatial_series,
               starting_time=0.0, rate=np.nan)

module_pos = nwbfile.create_processing_module(name='position', source=' ',
                                              description='position')

module_pos.add_container(pos)

data_names = ['cellImages', 'cellTraces', 'centroids', 'eventTimes']
mat_data = {}
mat_output_fpath = os.path.join(base_dir, 'm655_D11_S1_output.mat')
with File(mat_output_fpath, 'r') as file:
    for name in data_names:
        mat_data[name] = file['output'][name][:]


optical_channel = OpticalChannel('my channel', 'source', 'description',
                                 np.nan)

imaging_plane = nwbfile.create_imaging_plane(
    'my_imgpln', 'm655_D11_S1_output.mat', optical_channel,
    'description', 'imaging_device_1',
    600., '2.718', 'GFP', 'my favorite brain location',
    [], 4.0, 'manifold unit', 'A frame to refer to')


images_path = os.path.join(base_dir, 'm655_D11_S1.hdf5')
with File(images_path, 'r') as file:
    image_series = TwoPhotonSeries(name='test_iS', source='Ca2+ imaging example', dimension=[2],
                                   data=file['Data']['Images'][:], imaging_plane=imaging_plane,
                                   starting_frame=[0], starting_time=0.0, rate=5.0, scan_line_rate=np.nan,
                                   pmt_gain=np.nan)
nwbfile.add_acquisition(image_series)


mod = nwbfile.create_processing_module('rois', 'Ca2+ imaging example', 'example data module')
img_seg = ImageSegmentation('Ca2+ imaging example')
ps = img_seg.create_plane_segmentation('Ca2+ imaging example',
                                       'output from segmenting my favorite imaging plane',
                                       imaging_plane, 'my_planeseg', image_series)
mod.add_data_interface(img_seg)


for i, img_mask in enumerate(zip(mat_data['cellImages'])):
    pixel_mask = np.array(np.where(img_mask)).T
    ps.add_roi(str(i), pixel_mask, img_mask)


region = ps.create_roi_table_region('all', region=list(range(len(mat_data['cellImages']))))

roi_response = RoiResponseSeries('RoiResponseSeries', 'source', mat_data['cellTraces'],
                                 'lumens?', region, rate=5.0, starting_time=0.0)
fl = Fluorescence('source', roi_response_series=roi_response)
mod.add_data_interface(fl)

fname_out = 'm655_D11_S1.nwb'
with NWBHDF5IO(fname_out, 'w') as io:
    io.write(nwbfile)

with NWBHDF5IO(fname_out, 'r') as io:
    io.read()
