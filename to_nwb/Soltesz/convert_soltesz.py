import os

from h5py import File
from tqdm import tqdm
from datetime import datetime
import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.misc import SpikeUnit, UnitTimes
from pynwb.behavior import SpatialSeries, Position
from pynwb.form.backends.hdf5 import H5DataIO

from to_nwb.utils import pairwise
from to_nwb.general import PopulationSpikeTimes, CatCellInfo


def convert_file1(fpath, session_start_time,
                 session_description='simulated MEC and LEC data'):

    fname = os.path.split(fpath)[1]
    identifier = fname[:-4]
    institution = 'Stanford'
    lab = 'Soltesz'

    # extract data
    spike_units = []
    with File(fpath, 'r') as f:
        for cell_type in ('MPP', 'LPP'):
            spiketrain = f['Populations'][cell_type]['Vector Stimulus 0']['spiketrain']
            for i, (start, fin) in tqdm(enumerate(pairwise(spiketrain['Attribute Pointer'])),
                                        total=len(spiketrain['Attribute Pointer']),
                                        desc=cell_type):
                if not (start == fin):
                    UnitData = spiketrain['Attribute Value'][start:fin] / 1000
                    spike_units.append(SpikeUnit(name=cell_type + '{:05d}'.format(i),
                                                 times=UnitData,
                                                 unit_description=cell_type))

        ## Position
        x = f['Trajectory 0']['x']
        y = f['Trajectory 0']['y']
        rate = 1 / (f['Trajectory 0']['t'][1] - f['Trajectory 0']['t'][0]) * 1000

        pos_data = np.array([x, y]).T

    # write to NWB
    nwbfile = NWBFile(session_description, identifier,
                      session_start_time, datetime.now(),
                      institution=institution, lab=lab)

    rf_module = nwbfile.create_processing_module('receptive fields', 'spike times')

    spatial_series = SpatialSeries('Position', pos_data,
                                   reference_frame='NA',
                                   conversion=1 / 100.,
                                   resolution=0.1,
                                   starting_time=0.0,
                                   rate=rate)

    behav_ts = Position(spatial_series)
    unit_times = UnitTimes(spike_units, name='simulated cell spike data')

    rf_module.add_container(unit_times)
    rf_module.add_container(behav_ts)


def get_neuroh5_cell_data(fpath='../data/dentatenet_spikeout_Full_Scale_Control_7941551.bw.h5'):
    # process NeuroH5 file

    cell_index = []
    all_cell_types = []
    value_pointer = []
    value = []
    with File(fpath, 'r') as f:
        pops = f['Populations']
        for cell_type in pops:
            spike_struct = pops[cell_type]['Spike Events']['t']
            n = len(spike_struct['Cell Index'])

            all_cell_types += [cell_type] * n

            this_cell_index = spike_struct['Cell Index'][:]
            if cell_index:
                this_cell_index = this_cell_index + 1 + max(cell_index)
            cell_index += list(this_cell_index)

            this_value_pointer = spike_struct['Attribute Pointer'][:]
            if value_pointer:
                this_value_pointer = this_value_pointer[1:] + max(value_pointer)
            value_pointer += list(this_value_pointer)

            value += list(spike_struct['Attribute Value'][:])

        unique_cell_types, cell_type_indices = np.unique(all_cell_types, return_inverse=True)

    return {'cell_index': cell_index, 'unique_cell_types': unique_cell_types,
            'cell_type_indices': cell_type_indices,
            'value_pointer': value_pointer, 'value': value}


def write_nwb(cell_data, fpath='../data/soltesz_data.nwb', compress=True):
    """

    Parameters
    ----------
    cell_data: dict
        output of get_neuroh5_cell_data
    fpath: str
    compress: bool, optional
        if True, will compress all data. default=True

    Returns
    -------

    """
    if compress:
        cell_data = {key: H5DataIO(val, compress=True)
                     for key, val in cell_data.items()}

    fname = os.path.split(fpath)[0]
    f = NWBFile(file_name=fname,
                session_description=fname[:-3],
                identifier=fname[:-3],
                session_start_time=datetime.now(),
                lab='Soltesz',
                institution='Stanford')

    population_module = f.create_processing_module(name='spikes',
                                                   description='description')

    population_module.add_container(
        CatCellInfo(name='Cell Types',
                    values=cell_data['unique_cell_types'],
                    indices=cell_data['cell_type_indices'],
                    cell_index=cell_data['cell_index']))

    population_module.add_container(
        PopulationSpikeTimes(name='Population Spike Times',
                             cell_index=cell_data['cell_index'],
                             value=cell_data['value'],
                             pointer=cell_data['value_pointer']))

    with NWBHDF5IO(fpath, mode='w') as io:
        io.write()


def main():
    cell_data = get_neuroh5_cell_data(fpath='/Users/bendichter/Desktop/Soltesz/data/dentatenet_spikeout_Full_Scale_Control_7941551.bw.h5')
    write_nwb(cell_data, fpath='/Users/bendichter/Desktop/Soltesz/data/soltesz_data.nwb')


if __name__ == '__main__':
    main()
