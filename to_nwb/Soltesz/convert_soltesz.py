import os

from h5py import File
from tqdm import tqdm
from datetime import datetime
import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position

from ..utils import check_module

import sys


def read_ragged_array(struct, x):

    if 'Cell Index' in struct:
        x = np.where(struct['Cell Index'][:] == x)[0]

    x = int(x)

    start = struct['Attribute Pointer'][x]
    stop = struct['Attribute Pointer'][x+1]

    return struct['Attribute Value'][start:stop].astype(float)


def get_neuroh5_cell_data(f):

    pops = f['Populations']
    for cell_type in pops:
        spike_struct = pops[cell_type]['Vector Stimulus 100']['spiketrain']
        for pop_id in spike_struct['Cell Index']:
            spike_times = read_ragged_array(spike_struct, pop_id) / 1000
            yield {'pop_id': int(pop_id), 'spike_times': spike_times, 'cell_type': cell_type}


def write_position(nwbfile, f, name='Trajectory 100'):
    obj = f[name]
    behavior_mod = check_module(nwbfile, 'behavior')

    spatial_series = SpatialSeries('Position', data=np.array([obj['x'], obj['y']]).T,
                                   reference_frame='NA',
                                   conversion=1 / 100.,
                                   resolution=np.nan,
                                   rate=1 / np.diff(obj['t'][:2]) * 1000)

    behavior_mod.add_data_interface(Position(spatial_series))

    return behavior_mod


def neuroh5_to_nwb(fpath='/Users/bendichter/Desktop/Soltesz/data/DG_PP_spikes_101718.h5', out_path=None):

    if out_path is None:
        out_path = fpath[:-3] + '.nwb'

    fname = os.path.split(fpath)[1]
    identifier = fname[:-4]

    nwbfile = NWBFile(session_description='session_description',
                      identifier=identifier,
                      session_start_time=datetime.now().astimezone(),
                      institution='Stanford University', lab='Soltesz')

    with File('fpath', 'r') as f:
        write_position(nwbfile, f)

        nwbfile.add_unit_column('cell_type', 'cell type')
        nwbfile.add_unit_column('pop_id', 'cell number within population')

        for unit_dict in tqdm(get_neuroh5_cell_data(fpath),
                              total=38000+34000,
                              desc='reading cell data'):
            nwbfile.add_unit(**unit_dict)

    with NWBHDF5IO(out_path, 'w') as io:
        io.write(nwbfile)


def main(argv):
    neuroh5_to_nwb('/Users/bendichter/Desktop/Soltesz/data/DG_PP_spikes_101718.h5')


if __name__ == "__main__":
    main(sys.argv[1:])
