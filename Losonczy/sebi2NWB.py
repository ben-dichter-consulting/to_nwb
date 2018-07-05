# coding: utf-8

# In[174]:


import os
import sys
from datetime import datetime

import h5py
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import Position, BehavioralTimeSeries, BehavioralEvents
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb.ophys import OpticalChannel, TwoPhotonSeries, Fluorescence, DfOverF, ImageSegmentation

from neuroscope import get_channel_groups

# Losonczy Imports
from lab.misc import lfp_helpers as lfph
from lab.misc.auto_helpers import get_element_size_um, get_prairieview_version
from lab.classes.dbclasses import dbExperiment


# TODO throughout, replace source fields with appropriate info

def get_position(region):

    if region == 'CA1':
        return [2.1, 1.5, 1.2]
    else:
        return [np.nan, np.nan, np.nan]


# Add LFP (to acquisitions, though this is already downsampled and converted from rhd we for now treat this as raw)

def add_LFP(nwbfile, expt, count=1, region='CA1'):
    eeg_local = [x for x in os.listdir(expt.LFPFilePath()) if x.endswith('.eeg')][0]
    eeg_file = os.path.join(expt.LFPFilePath(), eeg_local)
    eeg_base = eeg_file.replace('.eeg', '')
    eeg_dict = lfph.loadEEG(eeg_base)

    lfp_xml_fpath = eeg_base + '.xml'
    channel_groups = get_channel_groups(lfp_xml_fpath)
    lfp_channels = channel_groups[0]
    lfp_fs = eeg_dict['sampeFreq']
    nchannels = eeg_dict['nChannels']

    lfp_signal = eeg_dict['EEG'][lfp_channels].T

    device_name = 'LFP_Device_{}'.format(count)
    device = nwbfile.create_device(device_name, source='SOURCE')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        source=lfp_xml_fpath,
        description=device_name,
        device=device,
        location=region)

    x, y, z = get_position(region)

    for channel in channel_groups[0]:
        nwbfile.add_electrode(channel,
                              x, y, z,  # position?
                              imp=np.nan,
                              location=region,
                              filtering='See lab.misc.lfp_helpers.ConvertFromRHD',
                              description='lfp electrode {}'.format(channel),
                              group=electrode_group)

    lfp_table_region = nwbfile.create_electrode_table_region(list(range(len(lfp_channels))),
                                                             'lfp electrodes')

    # TODO add conversion field for moving to V
    # TODO figure out how to link lfp data (zipping seems kludgey)
    lfp_elec_series = ElectricalSeries(name='LFP',
                                       source='SOURCE',
                                       data=H5DataIO(lfp_signal,
                                                     compression='gzip'),
                                       electrodes=lfp_table_region,
                                       conversion=np.nan,
                                       starting_time=0.0,
                                       rate=lfp_fs,
                                       resolution=np.nan)

    nwbfile.add_acquisition(LFP(source='SOURCE', electrical_series=lfp_elec_series))


def add_imaging(nwbfile, expt, z_spacing=25., device='2P Microscope', location='CA1',
                indicator='GCaMP6f', excitation_lambda=920.):

    color_dict = {'Ch1': 'Red', 'Ch2': 'Green'}
    # Emissions for mCherry and GCaMP
    # TODO make this more flexible
    emission = {'Ch1': 640., 'Ch2': 530.}

    ch_names = ['Ch1', 'Ch2']

    optical_channels = []
    for ch_name in ch_names:

        optical_channel = OpticalChannel(
            name=ch_name,
            source='SOURCE',
            description=color_dict[ch_name],
            emission_lambda=emission[ch_name])

        optical_channels.append(optical_channel)

    h5_folder = os.path.dirname('/Users/bendichter/Desktop/Losonczy/from_sebi/TSeries-05042017-001/')
    h5_file = [x for x in os.listdir(h5_folder) if x.endswith('h5')][0]
    h5_path = os.path.join(h5_folder, h5_file)

    pv_xml = os.path.join(h5_folder, os.path.basename(h5_folder) + '.xml')
    pv_version = get_prairieview_version(pv_xml)
    [y_um, x_um] = get_element_size_um(pv_xml, pv_version)[-2:]

    elem_size_um = [z_spacing, y_um, x_um]

    # TODO allow for flexibility in setting device, excitation, indicator, location
    # TODO nwb-schema issue #151 needs to be resolved so we can actually use imaging data size
    imaging_plane = nwbfile.create_imaging_plane(
        name='Imaging Data', source='SOURCE',
        optical_channel=optical_channels,
        description='imaging data for both channels',
        device=device, excitation_lambda=excitation_lambda,
        imaging_rate=str(1 / expt.frame_period()), indicator=indicator,
        location=location,
        conversion=1.0,  # Should actually be elem_size_um
        unit='um')

    with h5py.File(h5_path, 'r') as f:
        all_imaging_data = f['imaging'][:10]
        channel_names = f['imaging'].attrs['channel_names']

    # t,z,y,x,c -> t,x,y,z,c
    all_imaging_data = np.swapaxes(all_imaging_data, 1, 3)

    # t,x,y,z,c -> c,t,(x,y,z)
    all_imaging_data = np.rollaxis(all_imaging_data, 4)

    for channel_name, imaging_data in zip(channel_names, all_imaging_data):
        # TODO parse env file to add power and pmt gain?
        image_series = TwoPhotonSeries(name='2p_Series_' + channel_name,
                                       source='SOURCE',
                                       dimension=expt.frame_shape()[:-1],
                                       data=H5DataIO(data=imaging_data, compression='gzip'),
                                       imaging_plane=imaging_plane,
                                       rate=1 / expt.frame_period(),
                                       starting_time=0.,
                                       description=channel_name)

        nwbfile.add_acquisition(image_series)


# Load in Behavior Data, store position, licking, and water reward delivery times
# TODO Include non-image synced data and check if there is imaging data before trying to add synced

def add_behavior(nwbfile, expt):

    bd = expt.behaviorData(imageSync=True)

    fs = 1 / expt.frame_period()

    behavior_module = nwbfile.create_processing_module(name='Behavior', source='SOURCE',
                                                       description='Data relevant to behavior')

    # Add Normalized Position

    pos = Position(source='SOURCE', name='Normalized Position')
    pos.create_spatial_series(name='Normalized Position', source='SOURCE',
                              data=bd['treadmillPosition'], reference_frame='0 is belt start',
                              conversion=0.001 * bd['trackLength'], rate=fs, starting_time=0.0)

    behavior_module.add_container(pos)

    # Add Licking

    licking = BehavioralTimeSeries(source='SOURCE', name='Licking')
    licking.create_timeseries(source='SOURCE', name='Licking', data=bd['licking'], starting_time=0.0,
                              rate=fs, description='1 if mouse licked during this imaging frame')

    behavior_module.add_container(licking)

    # Add Water Reward Delivery

    water = BehavioralTimeSeries(source='SOURCE', name='Water')
    water.create_timeseries(source='SOURCE', name='Water', data=bd['water'], starting_time=0.0,
                            rate=fs, description='1 if water was delivered during this imaging frame')

    behavior_module.add_container(water)

    # Add Lap Times

    laps = BehavioralEvents(source='SOURCE', name='Lap Starts')
    # TODO probably not best to have laps as data and timestamps here
    laps.create_timeseries(source='SOURCE', name='Lap Starts', data=bd['lap'],
                           timestamps=bd['lap'], description='Frames at which laps began')

    behavior_module.add_container(laps)


# ROI Utilities

def get_pixel_mask(roi):

    image_mask = get_image_mask(roi)
    inds = zip(*np.where(image_mask))
    out = [list(ind) + [1.] for ind in inds]
    return out


def get_image_mask(roi):
    return np.dstack([mask.toarray().T for mask in roi.mask])


def add_rois(nwbfile, module, expt):

    img_seg = ImageSegmentation(source='SOURCE')
    module.add_data_interface(img_seg)
    ps = img_seg.create_plane_segmentation(source='SOURCE', description='ROIs',
                                           imaging_plane=nwbfile.get_imaging_plane('Imaging Data'),
                                           name='Plane Segmentation')

    rois = expt.rois()
    for roi in rois:
        ps.add_roi(roi.label, get_pixel_mask(roi), get_image_mask(roi))

    return ps


        # TODO finish this!


def add_signals(module, expt, rt_region):

    fs = 1 / expt.frame_period()

    fluor = Fluorescence(source='SOURCE')
    sigs = expt.imagingData(dFOverF=None)
    fluor.create_roi_response_series(source='SOURCE', name='Fluorescence',
                                     data=sigs.squeeze(), rate=fs, unit='NA',
                                     rois=rt_region, starting_time=0.0)

    module.add_data_interface(fluor)


def add_dff(module, expt, rt_region):

    fs = 1 / expt.frame_period()

    fluor = DfOverF(source='SOURCE', name='DFF')
    sigs = expt.imagingData(dFOverF=None)
    fluor.create_roi_response_series(source='SOURCE', name='DFF',
                                     starting_time=0.0,
                                     data=sigs.squeeze(), rate=fs, unit='NA',
                                     rois=rt_region)

    module.add_data_interface(fluor)


def main(argv):
    # Lab-side read expts
    expt = dbExperiment(10304)
    expt.set_sima_path('/Users/bendichter/Desktop/Losonczy/from_sebi/TSeries-05042017-001/TSeries-05042017-001.sima')
    expt.tSeriesDirectory = '/Users/bendichter/Desktop/Losonczy/from_sebi/TSeries-05042017-001'
    expt.behavior_file = '/Users/bendichter/Desktop/Losonczy/from_sebi/TSeries-05042017-001/svr009_20170509113637.pkl'

    # Initialize NWBFile directly from experiment object metadata

    nwbfile = NWBFile(source='Losonczy Lab',  # required
                      session_description='{} experiment for mouse {}'.format(
                          expt.experimentType, expt.parent.mouse_name),  # required
                      identifier='{}'.format(expt.trial_id),  # required
                      session_start_time=expt.get('startTime'),  # required
                      file_create_date=datetime.now(),  # optional
                      experimenter=expt.project_name,  # optional
                      session_id='{}-{}-{}'.format(
                          expt.get('condition'), expt.get('day'), expt.get('session')),  # optional
                      institution='Columbia University',  # optional
                      lab='Losonczy Lab')  # optional

    add_imaging(nwbfile, expt)

    add_LFP(nwbfile, expt)

    add_behavior(nwbfile, expt)


    imaging_module = nwbfile.create_processing_module(name='im_analysis', source='SOURCE',
                                                      description='Data relevant to imaging')

    ps = add_rois(nwbfile, imaging_module, expt)
    """
    rt_region = ps.create_roi_table_region('all ROIs', region=list(range(len(expt.rois()))))

    add_signals(imaging_module, expt, rt_region)

    add_dff(imaging_module, expt, rt_region)

    """
    fout = '/Users/bendichter/Desktop/Losonczy/from_sebi/TSeries-05042017-001.nwb'
    print('writing...')

    with NWBHDF5IO(fout, 'w') as io:
        io.write(nwbfile)

    with NWBHDF5IO(fout) as io:
        io.read()

    # still to do:
    # Motion Corrections (just displacements?)
    # ROIs (finish)
    # Transients, Spikes
    # Place Fields
    # SWRs
    # Actually save to disk


if __name__ == "__main__":
    main(sys.argv[1:])
