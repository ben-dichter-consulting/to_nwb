import os
import argparse
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
from os import path

from h5py import File

from scipy.io import loadmat

from pynwb import NWBFile, TimeSeries, get_manager, load_namespaces,\
    get_class
from pynwb.misc import IntervalSeries
from pynwb.ecephys import ElectricalSeries
from pynwb.form.backends.hdf5 import HDF5IO

from chang.HTK import readHTK
from chang.utils import remove_duplicates

import pdb

"""
Convert ECoG to NWB
"""


def wav_path_key(path):
    num = path[path.find('Wav')+3:path.rfind('.')]
    return int(num[0]), int(num[1:])


def chang2nwb(blockpath, outpath=None, session_start_time=datetime(1900, 1, 1),
              session_description=None, identifier=None, use_anin4=False,
              ecog_format='mat', **kwargs):
    """

    Parameters
    ----------
    blockpath: str
    outpath: None | str
        if None, the blockname is used and is saved in the blockpath
    session_start_time: datetime.datetime
        default: datetime(1900, 1, 1)
    session_description: str
        default: blockname
    identifier: str
        default: blockname
    kwargs: dict
        passed to pynwb.NWBFile

    Returns
    -------

    """
    blockname = os.path.split(blockpath)[1]
    if identifier is None:
        identifier = blockname

    if session_description is None:
        session_description = blockname

    if outpath is None:
        outpath = blockpath + '.nwb'

    # Establish the assumptions about file paths
    mic_file = path.join(blockpath, 'Analog', 'ANIN1.htk')
    L_speaker_file = path.join(blockpath, 'Analog', 'ANIN2.htk')
    R_speaker_file = path.join(blockpath, 'Analog', 'ANIN3.htk')
    bad_time_file = path.join(blockpath, 'Artifacts', 'badTimeSegments.mat')
    elec_metadata_file = path.join(blockpath, 'elecs', 'TDT_elecs_all.mat')
    electrode_data_path = path.join(blockpath, 'RawHTK')
    ecog400_path = path.join(blockpath, 'ecog400', 'ecog.mat')
    mesh_path = path.join(blockpath, 'Meshes')

    if use_anin4:
        aux_file = path.join(blockpath, 'Analog', 'ANIN4.htk')

    # Get the paths to all HTK files and sort them
    htk_paths = sorted(glob.glob(path.join(electrode_data_path, '*.htk')),
                       key=wav_path_key)

    # Get metadata for all electrodes
    elecs_metadata = sio.loadmat(elec_metadata_file)
    elec_grp_xyz_coord = elecs_metadata['elecmatrix']
    anatomy = elecs_metadata['anatomy']
    elec_grp_loc = [str(x[3][0]) if len(x[3]) else "" for x in anatomy]
    elec_grp_type = [str(x[2][0]) for x in anatomy]
    elec_grp_long_name = [str(x[1][0]) for x in anatomy]
    elec_grp_device = [x[:x.find('Electrode')] for x in elec_grp_long_name]
    elec_grp_short_name = [str(x[0][0]) for x in anatomy]
    anatomy = {'loc': elec_grp_loc, 'type': elec_grp_type,
               'long_name': elec_grp_long_name, 'short_name': elec_grp_short_name,
               'device': elec_grp_device}
    elec_grp_df = pd.DataFrame(anatomy)

    n = len(elec_grp_long_name)
    if n < len(elec_grp_xyz_coord):
        coord = elec_grp_xyz_coord[:n]
    elif n == len(elec_grp_xyz_coord):
        coord = elec_grp_xyz_coord
    else:
        coord = elec_grp_xyz_coord
        for i in range(n - len(elec_grp_xyz_coord)):
            coord.append([np.nan, np.nan, np.nan])

    # Create the NWB file object
    nwbfile = NWBFile('source', session_description, identifier,
                      session_start_time, datetime.now(),
                      institution='University of California, San Francisco',
                      lab='Chang Lab', **kwargs)

    for device_name in remove_duplicates(elec_grp_device):
        device_data = elec_grp_df[elec_grp_df['device'] == device_name]
        # Create devices
        device = nwbfile.create_device(device_name, 'source')

        # Create electrode groups
        electrode_group = nwbfile.create_electrode_group(
            name=device_name + ' electrodes',
            source='source',
            description=device_name,
            location=device_data['type'].iloc[0],
            device=device
        )

        for idx, elec_data in device_data.iterrows():
            nwbfile.add_electrode(idx,
                                  float(coord[idx, 0]),
                                  float(coord[idx, 1]),
                                  float(coord[idx, 2]),
                                  imp=np.nan,
                                  location=elec_data['loc'],
                                  filtering='none',
                                  description=elec_data['short_name'],
                                  group=electrode_group)

        electrode_table_region = nwbfile.create_electrode_table_region(
            list(range(n)), 'all electrodes on device')

        # Read electrophysiology data from HTK files and add them to NWB file
        if ecog_format == 'htk':
            htk = readHTK(htk_paths[0])
            data = np.concatenate([readHTK(htk_paths[i], scale_s_rate=True)['data']
                                   for i in device_data.index.values]).T
            rate = htk['sampling_rate']

        elif ecog_format == 'mat':
            with File(ecog400_path, 'r') as f:
                data = f['ecogDS']['data'][:]
                rate = f['ecogDS']['sampFreq'][:].ravel()[0]

        ts_desc = "data generated from electrode group %s, sampled at %0.6f " \
                  "Hz" % (device_name, rate)

        ephys_ts = ElectricalSeries(device_name, "source", data,
                                    electrode_table_region, starting_time=0.0,
                                    rate=rate, description=ts_desc,
                                    conversion=0.001)
        nwbfile.add_acquisition(ephys_ts)

    # Add microphone recording from room
    mic_htk = readHTK(mic_file, scale_s_rate=True)
    nwbfile.add_acquisition(TimeSeries('ANIN1', 'microphone in room',
                                       mic_htk['data'][0],
                                       'audio unit', starting_time=0.0,
                                       rate=mic_htk['sampling_rate'],
                                       description="audio recording from "
                                                   "microphone in room"))

    # Add audio stimulus 1
    stim_htk = readHTK(L_speaker_file, scale_s_rate=True)
    nwbfile.add_stimulus(TimeSeries('ANIN2', 'the first stimulus source',
                                    stim_htk['data'][0], 'audio unit',
                                    starting_time=0.0,
                                    rate=stim_htk['sampling_rate'],
                                    description="audio stimulus 1"))

    # Add audio stimulus 2
    stim_htk = readHTK(R_speaker_file, scale_s_rate=True)
    nwbfile.add_stimulus(TimeSeries('ANIN3', "audio stimulus 2", stim_htk['data'][0],
                                    'audio unit', starting_time=0.0,
                                    rate=stim_htk['sampling_rate'],
                                    description='the second stimulus source'))

    if use_anin4:
        aux_htk = readHTK(aux_file, scale_s_rate=True)
        nwbfile.add_acquisition(TimeSeries('ANIN4', 'aux analog',
                                           aux_htk['data'][0],
                                           'aux unit', starting_time=0.0,
                                           rate=aux_htk['sampling_rate'],
                                           description="aux analog recording"))

    # Add bad time segments
    bad_time = sio.loadmat(bad_time_file)['badTimeSegments']
    ts_name = 'badTimeSegments'
    ts_source = bad_time_file      # this should be something more descriptive
    ts_desc = 'bad time segments'  # this should be something more descriptive
    bad_timepoints_ts = IntervalSeries(ts_name, ts_source, description=ts_desc)
    [bad_timepoints_ts.add_interval(start, stop) for start, stop in bad_time]

    if len(bad_time) > 0:
        nwbfile.add_raw_timeseries(bad_timepoints_ts)

    # import surface data
    load_namespaces('ecog.namespace.yaml')
    CorticalSurface = get_class('CorticalSurface', 'ecog')

    pial_files = glob.glob(path.join(mesh_path, '*pial.mat'))
    for pial_file in pial_files:
        matin = loadmat(pial_file)
        tri = matin['cortex']['tri'][0][0]
        vert = matin['cortex']['vert'][0][0]
        name = pial_file[pial_file.find('Meshes')+7:-4]
        nwbfile.add_acquisition(CorticalSurface(faces=tri, vertices=vert,
                                                name=name, source=pial_file))

    # Export the NWB file
    io = HDF5IO(outpath, get_manager(), mode='w')
    io.write(nwbfile)
    io.close()


def main():

    # Establish the assumptions about file paths
    raw = "RawHTK"
    analog = "Analog"
    artifacts = "Artifacts"
    meshes = "Meshes"
    desc = 'convert Raw ECoG data (in HTK) to NWB'
    epi = 'The following directories must be present: %s, %s, %s, and %s' % \
          (raw, analog, artifacts, meshes)

    parser = argparse.ArgumentParser(usage='%(prog)s data_dir out.nwb',
                                     description=desc, epilog=epi)
    parser.add_argument('blockpath', type=str,
                        help='the directory containing Raw ECoG data files')

    parser.add_argument('outfile', type=str, help='the path to the NWB file to write to')

    parser.add_argument('-s', '--scale', action='store_true', default=False,
                        help='specifies whether or not to scale sampling rate')

    args = parser.parse_args()

    chang2nwb(**args)


chang2nwb('/Users/bendichter/Desktop/Chang/data/EC61/EC61_B32',
          '/Users/bendichter/Desktop/Chang/data/EC61/EC61_B32.nwb')

#if __name__ == '__main__':
#    main()

