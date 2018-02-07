import os

from h5py import File
from tqdm import tqdm
from datetime import datetime
import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.misc import SpikeUnit, UnitTimes
from pynwb.behavior import SpatialSeries, Position

from .utils import pairwise


def convert_file1(fpath, session_start_time,
                 session_description='simulated MEC and LEC data'):

    fname = os.path.split(fpath)[1]
    identifier = fname[:-4]
    institution = 'Stanford'
    lab = 'Soltesz'
    source=fname[:-4]

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
                                                 unit_description=cell_type,
                                                 source=source))

        ## Position
        x = f['Trajectory 0']['x']
        y = f['Trajectory 0']['y']
        rate = 1 / (f['Trajectory 0']['t'][1] - f['Trajectory 0']['t'][0]) * 1000

        pos_data = np.array([x, y]).T

    # write to NWB
    nwbfile = NWBFile(source, session_description, identifier,
                      session_start_time, datetime.now(),
                      institution=institution, lab=lab)

    rf_module = nwbfile.create_processing_module('receptive fields', source, 'spike times')

    spatial_series = SpatialSeries('Position',
                                   source, pos_data,
                                   reference_frame='NA',
                                   conversion=1 / 100.,
                                   resolution=0.1,
                                   starting_time=0.0,
                                   rate=rate)

    behav_ts = Position(source, spatial_series)
    unit_times = UnitTimes(source, spike_units, name='simulated cell spike data')

    rf_module.add_container(unit_times)
    rf_module.add_container(behav_ts)
    
def get_neuroh5_cell_data(fname='dentatenet_spikeout_Full_Scale_Control_7941551.bw.h5'):
    # process NeuroH5 file
    fpath = os.path.join('../data', fname)

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

      unique_cell_types, cell_type_indices = np.unique(all_cell_types,
                                                       return_inverse=True)
      
  return {'cell_index': cell_index, 'unique_cell_types': unique_cell_types,
          'cell_type_indices': cell_type_indices,
          'value_pointer': value_pointer, 'value': value}

def write_nwb(data, fpath='../data/example12.nwb'):
    fname = os.path.split(fpath)[0]
    source = fname[:-3]
    f = NWBFile(file_name=fname,
                source=source,
                session_description=fname[:-3],
                identifier=fname[:-3],
                session_start_time=datetime.now(),
                lab='Soltesz',
                institution='Stanford')

    ns_path = "soltesz.namespace.yaml"
    ext_source = "soltesz.extensions.yaml"

    PopulationSpikeTimes = get_class('PopulationSpikeTimes', 'soltesz')
    CatCellInfo = get_class('CatCellInfo', 'soltesz')

    population_module = f.create_processing_module(name='0', source='source',
                                                   description='description')

    population_module.add_container(CatCellInfo(name='cell_types',
                                                source=source,
                                                values=data['unique_cell_types'],
                                                indices=data['cell_type_indices'],
                                                cell_index=data['cell_index']))

    population_module.add_container(PopulationSpikeTimes(name='population_spike_times',
                                                         source=source,
                                                         cell_index=data['cell_index'],
                                                         value=data['values'],
                                                         pointer=data['value_pointer']))

    io = NWBHDF5IO(fpath, mode='w')
    io.write(f)
    io.close()


