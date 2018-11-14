import argparse
import glob
import os
from datetime import datetime
from os import path

from scipy.io.wavfile import read as wavread

import numpy as np
import pandas as pd
import scipy.io as sio
from h5py import File
from pynwb import NWBFile, TimeSeries, get_manager, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.form.backends.hdf5 import H5DataIO
from pynwb.misc import IntervalSeries
from scipy.io import loadmat
from tqdm import tqdm

from pytz import timezone

from .HTK import readHTK
from ..utils import remove_duplicates

from ..extensions.time_frequency import HilbertSeries
from nwbext_ecog.ecog_manual import CorticalSurfaces

from .transcripts import parse, make_df

#ecog_ext = pynwb.extensions['ecog']
#Surface = ecog_ext.Surface
#CorticalSurfaces = ecog_ext.CorticalSurfaces


# get_manager must come after dynamic imports
manager = get_manager()


"""
Convert ECoG to NWB
"""


def get_analog(blockpath, num=1):
    wav_path = path.join(blockpath, 'Analog', 'analog' + str(num) + '.wav')
    if os.path.isfile(wav_path):
        rate, data = wavread(wav_path)
        return float(rate), np.array(data, dtype=float)
    htk_path = path.join(blockpath, 'Analog', 'ANIN' + str(num) + '.htk')
    if os.path.isfile(htk_path):
        return readHTK(htk_path, scale_s_rate=True)
    raise Exception('no analog path found for ' + str(num))


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


def add_cortical_surfaces(nwbfile, pial_files):

    names = []
    cortical_surface_object = CorticalSurfaces()
    for pial_file in pial_files:
        matin = loadmat(pial_file)
        if 'cortex' in matin:
            x = 'cortex'
        elif 'mesh' in matin:
            x = 'mesh'
        else:
            raise ValueError('Unknown structure of ' + pial_file + '.')
        tri = matin[x]['tri'][0][0] - 1
        vert = matin[x]['vert'][0][0]
        name = pial_file[pial_file.find('Meshes')+7:-4]
        names.append(name)
        cortical_surface_object.create_surface(faces=tri, vertices=vert, name=name)
    nwbfile.add_acquisition(cortical_surface_object)
    return nwbfile


def readhtks(htkpath, elecs=None, use_tqdm=True):
    if elecs is None:
        elecs = range(len(glob.glob(path.join(htkpath, 'Wav*.htk'))))
    data = []
    if use_tqdm:
        this_iter = tqdm(elecs, desc='reading electrodes')
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

    return rate, data


def chang2nwb(blockpath, outpath=None, session_start_time=None,
              session_description=None, identifier=None, anin4=False,
              ecog_format='htk', cortical_mesh=False, include_pitch=False,
              speakers=True, mic=True, mini=False, hilb=False, verbose=False,
              imaging_path=None, parse_transcript=False, **kwargs):
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
    cortical_mesh: str | bool (optional)
        False: (Default) cortical mesh is not saved
        'internal': cortical mesh is saved normally
        'external': cortical mesh is saved in an external file and a link is
            provided to that file. This is useful if you have multiple sessions for a single subject.
    include_pitch: bool (optional)
        add pitch data. Default: False
    speakers: bool (optional)
        Default: False
    mic: bool (optional)
        default: False
    mini: only save data stub
    hilb: bool
    kwargs: dict
        passed to pynwb.NWBFile

    Returns
    -------

    """

    basepath, blockname = os.path.split(blockpath)
    subject = get_subject(blockname)
    if identifier is None:
        identifier = blockname

    if session_description is None:
        session_description = blockname

    if outpath is None:
        outpath = blockpath + '.nwb'
    out_base_path = os.path.split(outpath)[0]

    if session_start_time is None:
        session_start_time = datetime(1900, 1, 1).astimezone(timezone('UTC'))

    if imaging_path is None:
        subj_imaging_path = path.join(basepath, 'imaging')
    else:
        subj_imaging_path = os.path.join(imaging_path, subject)

    # file paths
    bad_time_file = path.join(blockpath, 'Artifacts', 'badTimeSegments.mat')
    bad_channels_file = path.join(blockpath, 'Artifacts', 'badChannels.txt')
    lfp_path = path.join(blockpath, 'RawHTK')
    ecog400_path = path.join(blockpath, 'ecog400', 'ecog.mat')
    elec_metadata_file = path.join(subj_imaging_path, 'elecs', 'TDT_elecs_all.mat')
    hilbdir = path.join(blockpath, 'HilbAA_70to150_8band')
    mesh_path = path.join(subj_imaging_path, 'Meshes')
    pial_files = glob.glob(path.join(mesh_path, '*pial.mat'))
    if cortical_mesh and not len(pial_files):
        raise Warning('pial files not found')

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
    nwbfile = NWBFile(session_description, identifier,
                      session_start_time, datetime.now().astimezone(),
                      institution='University of California, San Francisco',
                      lab='Chang Lab', **kwargs)

    elec_grp_df['bad'] = np.zeros((len(elec_grp_df), ), dtype=bool)
    # I think bad channels is 1-indexed but I'm not sure
    if path.isfile(bad_channels_file) and os.stat(bad_channels_file).st_size:
        dat = pd.read_csv(bad_channels_file, header=None, delimiter='  ')
        bad_elecs_inds = dat.values.ravel() - 1
        elec_grp_df.loc[bad_elecs_inds, 'bad'] = True

    elec_counter = 0
    devices = remove_duplicates(elec_grp_device)
    devices = [x for x in devices if x not in ('NaN', 'Right', 'EKG')]

    nwbfile.add_electrode_column('bad', 'electrode identified as too noisy')
    for device_name in devices:
        device_data = elec_grp_df[elec_grp_df['device'] == device_name]
        # Create devices
        device = nwbfile.create_device(device_name)

        # Create electrode groups
        electrode_group = nwbfile.create_electrode_group(
            name=device_name + ' electrodes',
            description=device_name,
            location=device_data['type'].iloc[0],
            device=device
        )

        for idx, elec_data in device_data.iterrows():
            nwbfile.add_electrode(
                id=idx, x=float(coord[idx, 0]), y=float(coord[idx, 1]), z=float(coord[idx, 2]),
                imp=np.nan, location=elec_data['loc'], filtering='none', group=electrode_group,
                bad=elec_data['bad'])
            elec_counter += 1

    all_elecs = nwbfile.create_electrode_table_region(
        list(range(elec_counter)), 'all electrodes on brain')

    # Read electrophysiology data from HTK files and add them to NWB file
    if ecog_format == 'htk':
        if verbose:
            print('reading htk acquisition...', flush=True)
        rate, data = readhtks(lfp_path, lfp_elecs)
        data = data.squeeze()
        if verbose:
            print('done', flush=True)
        if ekg_elecs:
            ekg_data = readhtks(lfp_path, ekg_elecs)[1]
            ekg_ts = TimeSeries('EKG', H5DataIO(ekg_data, compression='gzip'),
                                rate=rate, unit='V', conversion=.001,
                                description='electrotorticography')
            nwbfile.add_acquisition(ekg_ts)

    elif ecog_format == 'mat':
        with File(ecog400_path, 'r') as f:
            data = f['ecogDS']['data'][:, lfp_elecs]
            rate = f['ecogDS']['sampFreq'][:].ravel()[0]

            if ekg_elecs:
                ekg_data = f['ecogDS']['data'][:, ekg_elecs]
    else:
        raise ValueError('unrecognized argument: ecog_format')

    ts_desc = "all Wav data"

    if mini:
        data = data[:2000]

    lfp_ts = ElectricalSeries(name='ECoG', data=H5DataIO(data, compression='gzip'),
                              electrodes=all_elecs, rate=rate, description=ts_desc,
                              conversion=0.001)
    lfp = LFP(electrical_series=lfp_ts)
    nwbfile.add_acquisition(lfp)

    if mic:
        # Add microphone recording from room
        fs, data = get_analog(blockpath, 1)
        nwbfile.add_acquisition(TimeSeries('microphone', data, 'audio unit', rate=fs,
                                           description="audio recording from microphone in room"))
    if speakers:
        fs, data = get_analog(blockpath, 2)
        # Add audio stimulus 1
        nwbfile.add_stimulus(TimeSeries('speaker 1', data, 'NA', rate=fs,
                                        description="audio stimulus 1"))

        # Add audio stimulus 2
        fs, data = get_analog(blockpath, 3)
        nwbfile.add_stimulus(TimeSeries('speaker 2', data, 'NA', rate=fs,
                                        description='the second stimulus source'))

    if anin4:
        fs, data = get_analog(blockpath, 4)
        nwbfile.add_acquisition(TimeSeries(anin4, data, 'aux unit', rate=fs,
                                           description="aux analog recording"))

    # Add bad time segments
    if os.path.exists(bad_time_file):
        bad_time = sio.loadmat(bad_time_file)['badTimeSegments']
        ts_name = 'badTimeSegments'
        ts_desc = 'bad time segments'  # this should be something more descriptive
        bad_timepoints_ts = IntervalSeries(ts_name, description=ts_desc)
        [bad_timepoints_ts.add_interval(start, stop) for start, stop in bad_time]

        if len(bad_time) > 0:
            nwbfile.add_acquisition(bad_timepoints_ts)

    if hilb:
        data, rate = readhtks(hilbdir)
        # you must have 1 or more of the following:
        #   data (analytic amplitude),
        #   real_data,
        #   imaginary_data,
        #   phase_data
        hs = HilbertSeries(name='hilbert_series', filter_centers=[1., 2., 3.],
                           filter_sigmas=[1., 2., 3.], data=data, rate=rate, electrodes=all_elecs)

        hilb_mod = nwbfile.create_processing_module(name='hilbert', description='na')
        hilb_mod.add_container(hs)

    if cortical_mesh == 'external':
        anat_fpath = path.join(out_base_path, subject + '_cortical_surface.nwbaux')
        if not os.path.isfile(anat_fpath):
            anat_nwbfile = NWBFile(
                session_description=subject + ' anatomy', identifier=subject + '_cortical_surface',
                session_start_time=datetime(1900, 1, 1).astimezone(timezone('UTC')))
            anat_nwbfile = add_cortical_surfaces(anat_nwbfile, pial_files)
            with NWBHDF5IO(anat_fpath, manager=manager, mode='w') as anat_io:
                anat_io.write(anat_nwbfile)
        anat_read_io = NWBHDF5IO(anat_fpath, manager=manager, mode='r')
        anat_nwbfile = anat_read_io.read()
        cortical_surfaces = anat_nwbfile.acquisition['cortical_surfaces']
        nwbfile.add_acquisition(cortical_surfaces)

    elif cortical_mesh == 'internal':
        nwbfile, surface_names = add_cortical_surfaces(nwbfile, pial_files)
    elif cortical_mesh is False:
        pass
    else:
        raise ValueError('bad value for cortical_mesh.')

    if parse_transcript:
        parseout = parse(blockpath, blockname)
        df = make_df(parseout, blockname, subject, align_pos=1)
        nwbfile.add_trial_column(
            'cv_transition', 'time from start to CV transition in seconds')
        nwbfile.add_trial_column(
            'speak', 'if True, subject is speaking. If False, subject is listening')
        nwbfile.add_trial_column('condition', 'syllable spoken')
        for _, row in df.iterrows():
            nwbfile.add_trial(
                start_time=row['start'], stop_time=row['stop'],
                cv_transition=row['align'] - row['start'],
                speak=row['mode'] == 'speak', condition=row['label'])

    if include_pitch:
        pass  # add pitch here

    # Export the NWB file
    with NWBHDF5IO(outpath, manager=manager, mode='w') as io:
        io.write(nwbfile)

    if cortical_mesh == 'external':
        anat_read_io.close()

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


if __name__ == '__main__':
    main()
