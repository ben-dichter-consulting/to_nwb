import os
from functools import partialmethod

from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.io import loadmat
from dateutil.parser import parse as dateparse
import pandas as pd

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.behavior import SpatialSeries, Position
from pynwb.ecephys import ElectricalSeries

from pynwb.form.backends.hdf5 import H5DataIO as gzip
gzip.__init__ = partialmethod(gzip.__init__, compression='gzip')

from to_nwb import neuroscope as ns

#from general import CatCellInfo, gzip

WRITE_ALL_LFPS = False

# get sessionInfo

fpath = '/Users/bendichter/dev/buzcode/exampleDataStructs/fbasename'

fpath_base, fname = os.path.split(fpath)

session_info_matin = loadmat(
    '/Users/bendichter/dev/buzcode/exampleDataStructs/20170505_396um_0um_merge.sessionInfo.mat',
    struct_as_record=True)
date_text = session_info_matin['sessionInfo']['Date'][0][0][0]


animal_info_matin = loadmat(
    os.path.join(fpath_base, fname + '.animalMetaData.mat'),
    struct_as_record=True)

keys = ('ID', 'strain', 'species', 'surgeryDate')
animal_info = {key: animal_info_matin['animal'][key][0][0][0] for key in keys}


session_description = 'mouse in open exploration and theta maze'
identifier = fname
institution = 'NYU'
lab = 'Buzsaki'

session_start_time = dateparse(date_text, yearfirst=True)


subject = Subject(subject_id=animal_info['ID'],
                  strain=animal_info['strain'],
                  species=animal_info['species'], source='source')

if 'DOB' in animal_info and type(animal_info['DOB']) is not str:
    subject.age = str(session_start_time - animal_info['DOB'])

source = fname
nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab, subject=subject)

all_ts = []

xml_filepath = os.path.join(fpath_base, fname + '.xml')

print(xml_filepath)
channel_groups = ns.get_channel_groups(xml_filepath)
shank_channels = ns.get_shank_channels(xml_filepath)
nshanks = len(shank_channels)
all_shank_channels = np.concatenate(shank_channels)
nchannels = sum(len(x) for x in channel_groups)
lfp_fs = ns.get_lfp_sampling_rate(xml_filepath)

lfp_channel = 0  # value taken from Yuta's spreadsheet

my_whl_file = Path(os.path.join(fpath_base, fname + '.whl'))

my_behavior_file = Path(os.path.join(fpath_base, fname + '.sessionInfo.mat'))

if my_whl_file.is_file():
    pos_df = ns.get_position_data(fpath_base, fname)
elif my_behavior_file.is_file():
    bI = loadmat(os.path.join(fpath_base, fname + '.behavior.mat'), struct_as_record=True)
    import pdb

    pdb.set_trace()
    # date_text = np.array2string(bI['behavior']['position'][0][0])

    # date_text = date_text[2:-2];
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)

print('done.')

print('setting up raw position data...', end='', flush=True)
# raw position sensors file
pos0 = nwbfile.add_acquisition(
    SpatialSeries('position sensor0',
                  'raw sensor data from sensor 0',
                  gzip(pos_df[['x0', 'y0']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))
all_ts.append(pos0)

pos1 = nwbfile.add_acquisition(
    SpatialSeries('position sensor1',
                  'raw sensor data from sensor 1',
                  gzip(pos_df[['x1', 'y1']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))
all_ts.append(pos1)
print('done.')

print('setting up electrodes...', end='', flush=True)
# shank electrodes
electrode_counter = 0
for shankn, channels in enumerate(shank_channels):
    device_name = 'shank{}'.format(shankn)
    device = nwbfile.create_device(device_name, fname + '.xml')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        source=fname + '.xml',
        description=device_name,
        device=device,
        location='unknown')
    for channel in channels:
        nwbfile.add_electrode(channel,
                              np.nan, np.nan, np.nan,  # position?
                              imp=np.nan,
                              location='unknown',
                              filtering='unknown',
                              description='electrode {} of shank {}, channel {}'.format(
                                  electrode_counter, shankn, channel),
                              group=electrode_group)

        if channel == lfp_channel:
            lfp_table_region = nwbfile.create_electrode_table_region(
                [electrode_counter], 'lfp electrode')

        electrode_counter += 1

# special electrodes
device_name = 'special'
device = nwbfile.create_device(device_name, fname + '.xml')
electrode_group = nwbfile.create_electrode_group(
    name=device_name + '_electrodes',
    source=fname + '.xml',
    description=device_name,
    device=device,
    location='unknown')
special_electrode_dict = {'ch_wait': 79, 'ch_arm': 78, 'ch_solL': 76,
                          'ch_solR': 77, 'ch_dig1': 65, 'ch_dig2': 68,
                          'ch_entL': 72, 'ch_entR': 71, 'ch_SsolL': 73,
                          'ch_SsolR': 70}
for name, num in special_electrode_dict.items():
    nwbfile.add_electrode(num,
                          np.nan, np.nan, np.nan,
                          imp=np.nan,
                          location='unknown',
                          filtering='unknown',
                          description=name,
                          group=electrode_group)
    nwbfile.create_electrode_table_region([electrode_counter], name)
    electrode_counter += 1

all_table_region = nwbfile.create_electrode_table_region(
    list(range(electrode_counter)), 'all electrodes')
print('done.')

# lfp
print('reading LFPs...', end='', flush=True)

my_lfp_file = Path(os.path.join(fpath_base, fname + '.lfp'))
my_eeg_file = Path(os.path.join(fpath_base, fname + '.eeg'))
lfp_file = 1
if my_lfp_file.is_file():
    lfp_file = os.path.join(fpath_base, fname + '.lfp')
elif my_eeg_file.is_file():
    lfp_file = os.path.join(fpath_base, fname + '.eeg')

if isinstance(lfp_file, str):

    # this needs to be rewritten to
    # 1) pull the number of channel (here hard coded as N = 80), from the XML
    # 2) load in chunks so you don't overwhelm the RAM

    all_channels = np.fromfile(lfp_file, dtype=np.int16).reshape(-1, 80)
    all_channels_lfp = all_channels[:, all_shank_channels]
    print('done.')

    if WRITE_ALL_LFPS:
        print('making ElectricalSeries objects for LFP...', end='', flush=True)
        all_lfp = nwbfile.add_acquisition(
            ElectricalSeries('all_lfp',
                             'lfp signal for all shank electrodes',
                             gzip(all_channels_lfp),
                             all_table_region,
                             conversion=np.nan,
                             starting_time=0.0,
                             rate=lfp_fs,
                             resolution=np.nan))
        all_ts.append(all_lfp)
        print('done.')

    lfp = nwbfile.add_acquisition(
        ElectricalSeries('lfp',
                         'signal used as the reference lfp',
                         gzip(all_channels[:, lfp_channel]),
                         lfp_table_region,
                         conversion=np.nan,
                         starting_time=0.0,
                         rate=lfp_fs,
                         resolution=np.nan))
    all_ts.append(lfp)

# create epochs corresponding to experiments/environments for the mouse
task_types = ['OpenFieldPosition_ExtraLarge', 'OpenFieldPosition_New_Curtain',
              'OpenFieldPosition_New', 'OpenFieldPosition_Old_Curtain',
              'OpenFieldPosition_Old', 'OpenFieldPosition_Oldlast']

module_behavior = nwbfile.create_processing_module(name='behavior',
                                                   source=source,
                                                   description=source)
for label in task_types:
    print('loading normalized position data for ' + label + '...', end='', flush=True)
    file = os.path.join(fpath, fname + '__' + label)

    matin = loadmat(file)
    tt = matin['twhl_norm'][:, 0]
    pos_data = matin['twhl_norm'][:, 1:3]

    spatial_series_object = SpatialSeries(name=label + ' spatial_series',
                                          source='position sensor0',
                                          data=gzip(pos_data),
                                          reference_frame='unknown',
                                          conversion=np.nan,
                                          resolution=np.nan,
                                          timestamps=gzip(tt))
    pos_obj = Position(source=source, spatial_series=spatial_series_object,
                       name=label + '_position')

    module_behavior.add_container(pos_obj)

    for i, window in enumerate(exp_times):
        nwbfile.create_epoch(start_time=window[0], stop_time=window[1],
                             tags=tuple(), description=label + '_' + str(i),
                             timeseries=all_ts + [spatial_series_object])
    print('done.')

## load celltypes
matin = loadmat(os.path.join(fpath_base, 'DG_all_6__UnitFeatureSummary_add.mat'),
                struct_as_record=False)['UnitFeatureCell'][0][0]

# taken from ReadMe
celltype_dict = {
    0: 'unknown',
    1: 'granule cells (DG) or pyramidal cells (CA3)  (need to use region info. see below.)',
    2: 'mossy cell',
    3: 'narrow waveform cell',
    4: 'optogenetically tagged SST cell',
    5: 'wide waveform cell (narrower, exclude opto tagged SST cell)',
    6: 'wide waveform cell (wider)',
    8: 'positive waveform unit (non-bursty)',
    9: 'positive waveform unit (bursty)',
    10: 'positive negative waveform unit'
}

region_dict = {3: 'CA3', 4: 'DG'}

this_file = matin.fname == fname
celltype_ids = matin.fineCellType.ravel()[this_file]
region_ids = matin.region.ravel()[this_file]
unit_ids = matin.unitID.ravel()[this_file]

celltype_names = []
for celltype_id, region_id in zip(celltype_ids, region_ids):
    if celltype_id == 1:
        if region_id == 3:
            celltype_names.append('pyramidal cell')
        elif region_id == 4:
            celltype_names.append('granule cell')
        else:
            raise Exception('unknown type')
    else:
        celltype_names.append(celltype_dict[celltype_id])

u_cats, indices = np.unique(celltype_names, return_inverse=True)

cci_obj = CatCellInfo(name='CellTypes',
                      source='DG_all_6__UnitFeatureSummary_add.mat',
                      values=u_cats, indices=indices,
                      cell_index=list(range(len(indices))))

ut_obj = ns.build_unit_times(fpath, fname)

module_spikes = nwbfile.create_processing_module('spikes', source=source,
                                                 description=source)

module_spikes.add_container(ut_obj)
module_spikes.add_container(cci_obj)

out_fname = fname + '.nwb'
print('writing NWB file...', end='', flush=True)
with NWBHDF5IO(out_fname, mode='w') as io:
    io.write(nwbfile, cache_spec=False)
print('done.')

print('testing read...', end='', flush=True)
# test read
with NWBHDF5IO(out_fname, mode='r') as io:
    io.read()
print('done.')
