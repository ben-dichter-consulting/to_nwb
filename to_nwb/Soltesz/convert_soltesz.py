import os

from h5py import File
from tqdm import tqdm
from datetime import datetime
import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position


def read_ragged_array(struct, x):

    if 'Cell Index' in struct:
        x = np.where(struct['Cell Index'][:]==x)[0]

    x = int(x)

    start = struct['Attribute Pointer'][x]
    stop = struct['Attribute Pointer'][x+1]

    return struct['Attribute Value'][start:stop].astype(float)


def get_neuroh5_cell_data(fpath):

    with File(fpath, 'r') as f:
        pops = f['Populations']
        for cell_type in pops:
            spike_struct = pops[cell_type]['Vector Stimulus 100']['spiketrain']
            for i, pop_id in enumerate(spike_struct['Cell Index']):
                spike_times = read_ragged_array(spike_struct, pop_id) / 100
                yield {'pop_id': int(pop_id), 'spike_times': spike_times, 'cell_type': cell_type}


fpath = '/Users/bendichter/Desktop/Soltesz/data/DG_PP_spikes_101718.h5'


fname = os.path.split(fpath)[1]
identifier = fname[:-4]

nwbfile = NWBFile(session_description='session_description',
                  identifier='identifier',
                  session_start_time=datetime.now().astimezone(),
                  institution='Stanford', lab='Soltesz')

# Position
trajectory = 'Trajectory 100'
with File(fpath, 'r') as f:
    x = f[trajectory]['x']
    y = f[trajectory]['y']
    rate = 1 / (f[trajectory]['t'][1] - f[trajectory]['t'][0]) * 1000

    pos_data = np.array([x, y]).T

behavior_mod = nwbfile.create_processing_module('behavior', 'holds behavior data')


spatial_series = SpatialSeries('Position', pos_data,
                               reference_frame='NA',
                               conversion=1 / 100.,
                               resolution=0.1,
                               rate=rate)

behavior_mod.add_data_interface(Position(spatial_series))

nwbfile.add_unit_column('cell_type', 'cell type')
nwbfile.add_unit_column('pop_id', 'cell number within population')
for unit_dict in tqdm(get_neuroh5_cell_data(fpath),
                      total=38000+34000,
                      desc='reading cell data'):
    nwbfile.add_unit(**unit_dict)

out_path = fpath[:-3] + '.nwb'
with NWBHDF5IO(out_path, 'w') as io:
    io.write(nwbfile)
