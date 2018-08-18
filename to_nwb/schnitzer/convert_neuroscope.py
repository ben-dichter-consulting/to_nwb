import os
import numpy as np
from to_nwb.neuroscope import get_lfp_sampling_rate, get_channel_groups

from pynwb.ecephys import ElectricalSeries, LFP
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from dateutil.parser import parse as parse_date

"""
Time simply increments by 1
"""

fpath_base = '/Users/bendichter/Desktop/Schnitzer/data/test1_171207_181558'
this_dir = fpath_base.split('/')[-1]
name, day, time = this_dir.split('_')

session_start_time = parse_date(day, yearfirst=True)

amp_xml_path = os.path.join(fpath_base, 'amplifier.xml')
amp_fs = get_lfp_sampling_rate(amp_xml_path)
nchannels = len(get_channel_groups(amp_xml_path)[0])


datas = ['amplifier', 'time', 'auxiliary', 'supply']
data_fpaths = {name: os.path.join(fpath_base, name + '.dat')
               for name in datas}

amp_data = np.fromfile(data_fpaths['amplifier'], dtype=np.int16).reshape(-1, nchannels)
time_data = np.fromfile(data_fpaths['time'], dtype=np.int32)
supply_data = np.fromfile(data_fpaths['supply'], dtype=np.int16)
ntt = len(amp_data)
aux_data = np.fromfile(data_fpaths['auxiliary'], dtype=np.int16).reshape(ntt, -1)

nwbfile = NWBFile(session_start_time=session_start_time, identifier=this_dir,
                  source=this_dir, session_description='unknown')

device = nwbfile.create_device(name='all_channels_device', source=' ')
group = nwbfile.create_electrode_group(name='all_channels_group',
                                       source=' ',
                                       description='all channels',
                                       device=device,
                                       location='unknown')

for i in range(nchannels):
    nwbfile.add_electrode(i,
                          np.nan, np.nan, np.nan,  # position
                          imp=np.nan,
                          location='unknown',
                          filtering='unknown',
                          description='electrode {}'.format(i),
                          group=group)
electrode_table_region = nwbfile.create_electrode_table_region(
    list(range(nchannels)), 'all electrodes')


electrical_series = ElectricalSeries(data=amp_data, starting_time=0.0,
                                     rate=amp_fs, units='unknown',
                                     electrodes=electrode_table_region,
                                     name='amp_data',
                                     source=data_fpaths['amplifier'])
nwbfile.add_acquisition(
    LFP(name='amp_data', source=data_fpaths['amplifier'],
        electrical_series=electrical_series))

nwbfile.add_acquisition(
    TimeSeries('auxiliary', source=data_fpaths['auxiliary'], data=aux_data,
               starting_time=0.0, rate=amp_fs))
nwbfile.add_acquisition(
    TimeSeries('supply', source=data_fpaths['supply'], data=supply_data,
               starting_time=0.0, rate=amp_fs))

out_fname = this_dir + '.nwb'
with NWBHDF5IO(out_fname, 'w') as io:
    io.write(nwbfile)

