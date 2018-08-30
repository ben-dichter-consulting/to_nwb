import argparse
import glob
import os
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
import scipy.io as sio
from h5py import File
from pynwb import NWBFile, TimeSeries, get_manager, load_namespaces, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.form.backends.hdf5 import H5DataIO
from pynwb.misc import IntervalSeries
from scipy.io import loadmat
from tqdm import tqdm

from .HTK import readHTK
from ..utils import remove_duplicates
from ..auto_class import get_class

from pynwb import register_class

"""
Convert ECoG to NWB
"""


def get_subject(blockname):
    return blockname[:blockname.find('_')]


def gen_htk_num(i):
    """Input 0-indexed channel number, output htk filename.
    Parameters
    ----------
    i: int
        zero-indexed channel number

    Returns
    -------
    str

    """
    return str(i//64+1) + str(np.mod(i, 64)+1)


def add_cortical_surface(nwbfile, pial_files):
    load_namespaces('ecog.namespace.yaml')
    CorticalSurface = get_class('CorticalSurface', 'ecog')

    names = []
    for pial_file in pial_files:
        matin = loadmat(pial_file)
        tri = matin['cortex']['tri'][0][0]
        vert = matin['cortex']['vert'][0][0]
        name = pial_file[pial_file.find('Meshes')+7:-4]
        names.append(name)
        nwbfile.add_acquisition(CorticalSurface(faces=tri, vertices=vert,
                                                name=name, source=pial_file))
    return nwbfile, names


def add_hilbert_series(nwbfile, hilbdir, electrodes):
    data, rate = readhtks(hilbdir)
    load_namespaces('/Users/bendichter/dev/to_nwb/to_nwb/chang/time_frequency.namespace.yaml')
    HilbertSeries = get_class('time_frequency', 'HilbertSeries')
    hs = HilbertSeries(name='hilbert_series', source=hilbdir, filter_centers=[1., 2., 3.],
                       filter_sigmas=[1., 2., 3.], data=data, rate=rate, electrodes=electrodes)

    hilb_mod = nwbfile.create_processing_module(name='hilbert', source='na', description='na')
    hilb_mod.add_container(hs)

    return nwbfile


def readhtks(htkpath, elecs=None, use_tqdm=True):
    if elecs is None:
        elecs = range(len(glob.glob(path.join(htkpath, 'Wav*.htk'))))
    data = []
    if use_tqdm:
        this_iter = tqdm(elecs)
    else:
        this_iter = elecs
    for i in this_iter:
        htk = readHTK(path.join(htkpath, 'Wav' + gen_htk_num(i) + '.htk'),
                      scale_s_rate=True)
        data.append(htk['data'])
    data = np.stack(data)
    if len(data.shape) == 3:
        data = data.transpose([2, 0, 1])

    rate = htk['sampling_rate']

    return data, rate


def chang2nwb(blockpath, outpath=None, session_start_time=datetime(1900, 1, 1),
              session_description=None, identifier=None, anin4=False,
              ecog_format='mat', cortical_mesh=False, include_pitch=False,
              speakers=True, mic=True, mini=False, hilb=False, **kwargs):
    """

    Parameters
    ----------
    blockpath: str
    outpath: None | str
        if None, output = [blockpath]/[blockname].nwb
    session_start_time: datetime.datetime
        default: datetime(1900, 1, 1)
    session_description: str
        default: blockname
    identifier: str
        default: blockname
    anin4: False | str
        Whether or not to convert ANIN4. ANIN4 is used as an extra channel for
        things like button presses, and is usually unused. If a string is
        supplied, that is used as the name of the timeseries.
    ecog_format: str
        ({'htk'}, 'mat')
    cortical_mesh: str, bool (optional)
        'internal': cortical mesh is saved normally
        'external': cortical mesh is saved in an external file and a link is
            provided to that file. This is useful if you have multiple sessions for a single subject.
        False: (Default) cortical mesh is not saved
    include_pitch: bool
        add pitch data. Default: False
    speakers: bool
    mic: bool
    mini: only save data stub
    kwargs: dict
        passed to pynwb.NWBFile

    Returns
    -------

    """
    manager = get_manager()

    basepath, blockname = os.path.split(blockpath)
    subject = get_subject(blockname)
    if identifier is None:
        identifier = blockname

    if session_description is None:
        session_description = blockname

    if outpath is None:
        outpath = blockpath + '.nwb'

    # file paths
    mic_file = path.join(blockpath, 'Analog', 'ANIN1.htk')
    L_speaker_file = path.join(blockpath, 'Analog', 'ANIN2.htk')
    R_speaker_file = path.join(blockpath, 'Analog', 'ANIN3.htk')
    bad_time_file = path.join(blockpath, 'Artifacts', 'badTimeSegments.mat')
    lfp_path = path.join(blockpath, 'RawHTK')
    ecog400_path = path.join(blockpath, 'ecog400', 'ecog.mat')
    elec_metadata_file = path.join(basepath, 'imaging', 'elecs', 'TDT_elecs_all.mat')
    mesh_path = path.join(blockpath, 'imaging', 'Meshes')
    aux_file = path.join(blockpath, 'Analog', 'ANIN4.htk')
    hilbdir = path.join(blockpath, 'HilbAA_70to150_8band')

    # Get metadata for all electrodes
    elecs_metadata = sio.loadmat(elec_metadata_file)
    elec_grp_xyz_coord = elecs_metadata['elecmatrix']
    anatomy = elecs_metadata['anatomy']
    elec_grp_loc = [str(x[3][0]) if len(x[3]) else "" for x in anatomy]
    elec_grp_type = [str(x[2][0]) for x in anatomy]
    elec_grp_long_name = [str(x[1][0]) for x in anatomy]

    if 'Electrode' in elec_grp_long_name[0]:
        elec_grp_device = [x[:x.find('Electrode')] for x in elec_grp_long_name]
    else:
        elec_grp_device = [''.join(filter(lambda y: not str.isdigit(y), x))
                           for x in elec_grp_long_name]

    elec_grp_short_name = [str(x[0][0]) for x in anatomy]

    lfp_elecs = [i for i, label in enumerate(elec_grp_short_name)
                 if label not in ('RT', 'EKG', 'NaN')]

    ekg_elecs = [i for i, label in enumerate(elec_grp_short_name)
                 if label == 'EKG']

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

    elec_counter = 0
    devices = remove_duplicates(elec_grp_device)
    devices = [x for x in devices if x not in ('NaN', 'Right', 'EKG')]
    for device_name in devices:
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
            elec_counter += 1

    all_elecs = nwbfile.create_electrode_table_region(
        list(range(elec_counter)), 'all electrodes on brain')

    # Read electrophysiology data from HTK files and add them to NWB file
    if ecog_format == 'htk':
        data, rate = readhtks(lfp_path, lfp_elecs)
        if ekg_elecs:
            ekg_data, _ = readhtks(lfp_path, ekg_elecs)

    elif ecog_format == 'mat':
        with File(ecog400_path, 'r') as f:
            data = f['ecogDS']['data'][:, lfp_elecs]
            rate = f['ecogDS']['sampFreq'][:].ravel()[0]

            if ekg_elecs:
                ekg_data = f['ecogDS']['data'][:, ekg_elecs]

    ts_desc = "all Wav data"

    if mini:
        data = data[:2000]

    lfp_ts = ElectricalSeries('LFP', "source", H5DataIO(data, compression='gzip'),
                              all_elecs, rate=rate, description=ts_desc,
                              conversion=0.001)
    nwbfile.add_acquisition(lfp_ts)

    ekg_ts = TimeSeries('EKG', 'source', H5DataIO(ekg_data, compression='gzip'),
                        rate=rate, unit='V', conversion=.001,
                        description='electrotorticography')
    nwbfile.add_acquisition(ekg_ts)

    if mic:
        # Add microphone recording from room
        mic_htk = readHTK(mic_file, scale_s_rate=True)
        nwbfile.add_acquisition(TimeSeries('microphone', 'microphone in room',
                                           mic_htk['data'][0],
                                           'audio unit', starting_time=0.0,
                                           rate=mic_htk['sampling_rate'],
                                           description="audio recording from "
                                                       "microphone in room"))
    if speakers:
        # Add audio stimulus 1
        stim_htk = readHTK(L_speaker_file, scale_s_rate=True)
        nwbfile.add_stimulus(TimeSeries('speaker 1', 'the first stimulus source',
                                        stim_htk['data'][0], 'audio unit',
                                        starting_time=0.0,
                                        rate=stim_htk['sampling_rate'],
                                        description="audio stimulus 1"))

        # Add audio stimulus 2
        stim_htk = readHTK(R_speaker_file, scale_s_rate=True)
        nwbfile.add_stimulus(TimeSeries('speaker 2', "audio stimulus 2", stim_htk['data'][0],
                                        'audio unit', starting_time=0.0,
                                        rate=stim_htk['sampling_rate'],
                                        description='the second stimulus source'))

    if anin4:
        aux_htk = readHTK(aux_file, scale_s_rate=True)
        nwbfile.add_acquisition(TimeSeries(anin4, 'aux analog',
                                           aux_htk['data'][0],
                                           'aux unit', starting_time=0.0,
                                           rate=aux_htk['sampling_rate'],
                                           description="aux analog recording"))

    # Add bad time segments
    if os.path.exists(bad_time_file):
        bad_time = sio.loadmat(bad_time_file)['badTimeSegments']
        ts_name = 'badTimeSegments'
        ts_source = bad_time_file      # this should be something more descriptive
        ts_desc = 'bad time segments'  # this should be something more descriptive
        bad_timepoints_ts = IntervalSeries(ts_name, ts_source, description=ts_desc)
        [bad_timepoints_ts.add_interval(start, stop) for start, stop in bad_time]

        if len(bad_time) > 0:
            nwbfile.add_raw_timeseries(bad_timepoints_ts)

    pial_files = glob.glob(path.join(mesh_path, '*pial.mat'))
    if cortical_mesh == 'external':
        anat_fpath = path.join(basepath, subject + '_cortical_surface.nwbaux')
        anat_nwbfile = NWBFile(source='',
                               session_description='',
                               identifier=subject + '_cortical_surface',
                               session_start_time=datetime(1900, 1, 1))  # placeholder since this argument is required
        anat_nwbfile, pial_names = add_cortical_surface(anat_nwbfile, pial_files)
        with NWBHDF5IO(anat_fpath, manager=manager, mode='w') as anat_io:
            anat_io.write(anat_nwbfile)

        anat_nwbfile = NWBHDF5IO(anat_fpath, manager=manager, mode='r').read()
        for pial_name in pial_names:
            surface_objects = anat_nwbfile.get_acquisition(pial_name)
            nwbfile.add_acquisition(surface_objects)

    elif cortical_mesh == 'internal':
        nwbfile = add_cortical_surface(nwbfile, pial_files)
    elif cortical_mesh is False:
        pass
    else:
        raise ValueError('bad value for cortical_mesh.')

    if include_pitch:
        pass  # add pitch here

    if hilb:
        nwbfile = add_hilbert_series(nwbfile, hilbdir, all_elecs)


    # Export the NWB file
    with NWBHDF5IO(outpath, manager=manager, mode='w') as io:
        io.write(nwbfile)

    # read check
    with NWBHDF5IO(outpath, manager=manager, mode='r') as io:
        io.read()


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


chang2nwb('/Users/bendichter/Desktop/Chang/data/EC169/EC169_B7')

if __name__ == '__main__':
    main()

