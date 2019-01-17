import os
import pickle
import re
from datetime import datetime
from glob import glob

import numpy as np
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from tqdm import tqdm

run_dir = '/Users/bendichter/Desktop/Poirazi/data/DATA_Ben'
session_start_time = datetime(2017, 4, 15, 12, tzinfo=tzlocal())
description = 'description of session'
identifier = 'session_id'


def natural_key(text):
    # Key used for natural ordering: orders files correctly even if numbers are not zero-padded
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]


# setup NWB file
nwbfile = NWBFile(session_description=description,
                  identifier=identifier,
                  session_start_time=session_start_time)
nwbfile.add_unit_column('cell_type', 'cell type')
nwbfile.add_unit_column('cell_type_id', 'integer index within each cell type')

# convert continuous data (1 compartment per cell)
mp_data = []
for dat_file in tqdm(sorted(glob(os.path.join(run_dir, '*dat')), key=natural_key),
                     desc='reading .dat files'):
        mp_data.append(np.loadtxt(dat_file))
mp_data = np.column_stack(mp_data)
ts = TimeSeries('membrane_potential', mp_data, unit='mV', rate=10000.)
nwbfile.add_acquisition(ts)

# convert spike data

for spk_file in sorted(glob(os.path.join(run_dir, 'spiketimes*'))):
    with open(spk_file, 'rb') as file:
        data = pickle.load(file)
    cell_type = spk_file.split('_')[-2]
    for row in data:
        cell_id = row[0]
        spikes = np.array(row[1:], dtype=np.float) / 1000
        nwbfile.add_unit(spike_times=spikes, cell_type=cell_type, cell_type_id=cell_id)

print(nwbfile.units['cell_type'].data)
print(nwbfile.units.get_unit_spike_times(21))

with NWBHDF5IO(run_dir + '.nwb', 'w') as io:
    io.write(nwbfile)
