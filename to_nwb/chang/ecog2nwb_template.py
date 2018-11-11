from datetime import datetime

import numpy as np
import pandas as pd
from nwbext_ecog.ecog_manual import CorticalSurfaces
from pynwb import NWBFile, TimeSeries, NWBHDF5IO, get_manager
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.file import Subject
from pynwb.misc import IntervalSeries
from pytz import timezone
from scipy.io.wavfile import read as wavread

# get_manager must come after dynamic imports
manager = get_manager()

external_cortical_mesh = True


def add_cortical_surfaces(nwbfile):
    cortical_surfaces = CorticalSurfaces()
    for name in ('a', 'b', 'c'):
        vertices = np.random.randn(10, 3)
        faces = np.random.randint(0, 9, (15, 3))
        cortical_surfaces.create_surface(name=name, faces=faces, vertices=vertices)
    nwbfile.add_acquisition(cortical_surfaces)

    return nwbfile


nwbfile = NWBFile('session description', 'session identifier',
                  datetime.now().astimezone(), institution='UCSF',
                  lab='Chang Lab')

nwbfile.subject = Subject(species='homo sapiens', age='PY21', sex='M')

# electrodes
devices = ['a', 'a', 'a', 'b', 'b', 'b']
locations = ['a location', 'b location']
udevices, inds = np.unique(devices, return_inverse=True)
groups = []
for device_name, location in zip(udevices, locations):
    # Create devices
    device = nwbfile.create_device(device_name)

    # Create electrode groups
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        description=device_name,
        location=location,
        device=device)
    groups.append(electrode_group)

nwbfile.add_electrode_column('bad', 'whether the electrode is too noisy to use')

electrodes_df = pd.DataFrame(
    {'location': ['c', 'c', 'c', 'd', 'd', 'd'],
     'group': np.array(groups)[inds],
     'x': [np.nan] * 6,
     'y': [np.nan] * 6,
     'z': [np.nan] * 6,
     'imp': [np.nan] * 6,
     'filtering': ['none'] * 6,
     'bad': [False] * 5 + [True]}
)

for _, row in electrodes_df.iterrows():
    nwbfile.add_electrode(**{label: row[label] for label in electrodes_df})

all_elecs = nwbfile.create_electrode_table_region(
        list(range(len(electrodes_df))), 'all electrodes')


# ECoG signal
lfp_signal = np.random.randn(1000, 64)
lfp_ts = ElectricalSeries('LFP', lfp_signal, all_elecs, rate=3000.,
                          description='lfp_signal', conversion=0.001)

lfp = LFP(electrical_series=lfp_ts)

nwbfile.add_acquisition(lfp)

# Trials
# optional columns
nwbfile.add_trial_column('condition', 'condition of task')
nwbfile.add_trial_column('response_latency', 'in seconds')
nwbfile.add_trial_column('response', 'y is yes, n is no')
nwbfile.add_trial_column('bad', 'whether a trial is bad either because of '
                                'artifact or bad performance')

trials_df = pd.DataFrame({'start_time': [1., 2., 3.],
                          'stop_time': [1.5, 2.5, 3.5],
                          'condition': ['a', 'b', 'c'],
                          'response_latency': [.3, .26, .31],
                          'response': ['y', 'n', 'y'],
                          'bad': [False, False, True]})

for _, row in trials_df.iterrows():
    nwbfile.add_trial(**{label: row[label] for label in trials_df})

# print(nwbfile.trials.to_dataframe())

# bad times
bad_times_data = [[5.4, 6.],
                  [10.4, 11.]]  # in seconds
bad_times = IntervalSeries(name='bad_times')
for start, stop in bad_times_data:
    bad_times.add_interval(start, stop)
nwbfile.add_acquisition(bad_times)

# Create units table for neurons from micro-array recordings
single_electrode_regions = [
    nwbfile.create_electrode_table_region([i], 'electrode i')
    for i in range(len(electrodes_df))]

all_spike_times = [[1., 2., 3., 4.],
                   [2., 3., 4.],
                   [0.5, 1., 4., 10., 15.]]

all_electrodes = ((0,), (0,), (1,))

waveform_means = [np.random.randn(30, 1) for _ in range(3)]

for spike_times, electrodes, waveform_mean in \
        zip(all_spike_times, all_electrodes, waveform_means):
    nwbfile.add_unit(spike_times=spike_times,
                     electrodes=electrodes,
                     waveform_mean=waveform_mean)

# analog data
# microphone data
# Be careful! This might contain identifying information
mic_path = '/Users/bendichter/Desktop/Chang/video_abstract/word_emphasis.wav'
mic_fs, mic_data = wavread(mic_path)
nwbfile.add_acquisition(
    TimeSeries('microphone', mic_data, 'audio unit', rate=float(mic_fs),
               description="audio recording from microphone in room")
)
# all analog data can be added like the microphone example (speaker, button press, etc.)
spk_path = '/Users/bendichter/Desktop/Chang/video_abstract/word_emphasis.wav'
spk_fs, spk_data = wavread(spk_path)
nwbfile.add_stimulus(
    TimeSeries('speaker1', spk_data, 'audio unit', rate=float(spk_fs),
               description="speaker recording")
)

# cortical surfaces
if not external_cortical_mesh:
    nwbfile = add_cortical_surfaces(nwbfile)
else:
    anat_fpath = 'S1.nwbaux'
    anat_nwbfile = NWBFile(
        session_description='session description', identifier='S1',
        session_start_time=datetime(1900, 1, 1).astimezone(timezone('UTC')))
    anat_nwbfile = add_cortical_surfaces(anat_nwbfile)
    with NWBHDF5IO(anat_fpath, manager=manager, mode='w') as anat_io:
        anat_io.write(anat_nwbfile)
    anat_read_io = NWBHDF5IO(anat_fpath, manager=manager, mode='r')
    anat_nwbfile = anat_read_io.read()
    cortical_surfaces = anat_nwbfile.acquisition['cortical_surfaces']
    nwbfile.add_acquisition(cortical_surfaces)

fout_path = 'ecog_example.nwb'
with NWBHDF5IO(fout_path, manager=manager, mode='w') as io:
    io.write(nwbfile)

# test read
with NWBHDF5IO(fout_path, 'r') as io:
    io.read()

if external_cortical_mesh:
    anat_read_io.close()
