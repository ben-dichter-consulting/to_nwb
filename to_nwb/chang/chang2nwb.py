from __future__ import print_function

import argparse
import glob
import os
from datetime import datetime
from os import path

import numpy as np
import pandas as pd
import scipy.io as sio
from h5py import File
from nwbext_ecog import CorticalSurfaces, ECoGSubject
from pynwb import NWBFile, TimeSeries, get_manager, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries
from pynwb.behavior import BehavioralTimeSeries
from pynwb.image import RGBImage, RGBAImage, GrayscaleImage
from pynwb.base import Images
from pynwb.misc import DecompositionSeries
from pynwb.file import Subject
from pytz import timezone
from scipy.io import loadmat
from scipy.io.wavfile import read as wavread
from scipy.misc import imread
from tqdm import tqdm

from hdmf.backends.hdf5 import H5DataIO
from hdmf.data_utils import DataChunkIterator

from .HTK import readHTK
from .transcripts import parse, make_df, create_transcription
from ..utils import remove_duplicates
from ..tdt import load_wavs, load_anin


# get_manager must come after dynamic imports
manager = get_manager()

raw_htk_paths = ['/data_store1/human/prcsd_data', '/data_store0/human/HTK_raw']
tdt_data_path = '/data_store0/human/HTK_raw'
IMAGING_PATH = '/data_store2/imaging/subjects'
hilb_dir = '/userdata/bdichter/from_jesse/'
transcript_path = '/userdata/jliu/data/human/EC118/lfp'


"""
Convert ECoG to NWB
"""


def find_ekg_elecs(elec_metadata_file):
    elec_grp_df, coord = read_electrodes(elec_metadata_file)
    return np.where(elec_grp_df['device'] == 'EKG')[0]


def add_ekg(nwbfile, ecog_path, ekg_elecs):
    if os.path.split(ecog_path)[1] == 'RawHTK':
        rate, data = readhtks(ecog_path, ekg_elecs)
    elif os.path.split(ecog_path)[1] == 'ecog.mat':
        with File(ecog_path, 'r') as f:
            data = f['ecogDS']['data'][:, ekg_elecs]
            rate = f['ecogDS']['sampFreq'][:].ravel()[0]
    elif os.path.split(ecog_path)[1] == 'raw.mat':
        rate, data = load_wavs(ecog_path, ekg_elecs)

    ekg_ts = TimeSeries('EKG', H5DataIO(data, compression='gzip'), unit='V',
                        rate=rate, conversion=.001, description='electrotorticography')
    nwbfile.add_acquisition(ekg_ts)


def add_images_to_subject(subject, subject_image_list):
    images = Images(name='images', description="images of subject's brain")
    for image_path in subject_image_list:
        image_name = os.path.split(image_path)[1]
        image_name = os.path.splitext(image_name)[0]
        image_data = imread(image_path)
        kwargs = {'data': image_data, 'name': image_name}
        if len(image_data.shape) == 2:
            image = GrayscaleImage(**kwargs)
        elif image_data.shape[2] == 3:
            image = RGBImage(**kwargs)
        elif image_data.shape[2] == 4:
            image = RGBAImage(**kwargs)

        images.add_image(image)
    subject.images = images
    return subject


def load_pitch(blockpath):
    blockname = os.path.split(blockpath)[1]
    pitch_path = os.path.join(blockpath, 'pitch_' + blockname + '.mat')
    matin = loadmat(pitch_path)
    data = matin['pitch'].ravel()
    fs = 1 / matin['dt'][0, 0]
    return fs, data


def load_intensity(blockpath):
    blockname = os.path.split(blockpath)[1]
    pitch_path = os.path.join(blockpath, 'intensity_' + blockname + '.mat')
    matin = loadmat(pitch_path)
    data = matin['intensity'].ravel()
    fs = 1 / matin['dt'][0, 0]
    return fs, data


def get_analog(blockpath, num=1):
    """
    Load analog data. Try:
    1) analog[num].wav
    2) ANIN[num].htk
    3) Extracting from raw.mat
    Parameters
    ----------
    blockpath: str
    num: int

    Returns
    -------
    fs, data

    """
    wav_path = path.join(blockpath, 'Analog', 'analog' + str(num) + '.wav')
    if os.path.isfile(wav_path):
        rate, data = wavread(wav_path)
        return float(rate), np.array(data, dtype=float)
    htk_path = path.join(blockpath, 'Analog', 'ANIN' + str(num) + '.htk')
    if os.path.isfile(htk_path):
        htk_out = readHTK(htk_path, scale_s_rate=True)
        return htk_out['sampling_rate'], htk_out['data'].ravel()
    blockname = os.path.split(blockpath)[1]
    subject_id = get_subject_id(blockname)
    raw_fpath = os.path.join(raw_htk_paths[0], subject_id, blockname, 'raw.mat')
    if os.path.isfile(raw_fpath):
        return load_anin(raw_fpath, num)
    print('no analog path found for ' + str(num))
    return None, None


def get_subject_id(blockname):
    return blockname[:blockname.find('_')]


def gen_htk_num(i, n=64):
    """Input 0-indexed channel number, output htk filename.
    Parameters
    ----------
    i: int
        zero-indexed channel number
    n: int
        number of channels per wav

    Returns
    -------
    str

    """
    return str(i//n+1) + str(np.mod(i, n)+1)


def auto_ecog(blockpath, ecog_elecs, verbose=False):
    basepath, blockname = os.path.split(blockpath)
    subject_id = get_subject_id(blockname)
    for htk_blockpath in [blockpath] + [os.path.join(x, subject_id, blockname) for x in raw_htk_paths]:
        htk_path = os.path.join(htk_blockpath, 'RawHTK')
        if os.path.exists(htk_path):
            if verbose:
                print('reading htk acquisition...', flush=True)
            fs, data = readhtks(htk_path, ecog_elecs)
            data = data.squeeze()
            if verbose:
                print('done', flush=True)
            return fs, data, htk_path
        raw_fpath = os.path.join(htk_blockpath, 'raw.mat')
        if os.path.exists(raw_fpath):
            fs, data = load_wavs(raw_fpath)
            return fs, data, raw_fpath


def create_cortical_surfaces(pial_files, subject_id):

    if not len(pial_files):
        return None

    names = []
    cortical_surfaces = CorticalSurfaces()
    for pial_file in pial_files:
        matin = loadmat(pial_file)
        if 'cortex' in matin:
            x = 'cortex'
        elif 'mesh' in matin:
            x = 'mesh'
        else:
            Warning('Unknown structure of ' + pial_file + '.')
            continue

        tri = matin[x]['tri'][0][0] - 1
        vert = matin[x]['vert'][0][0]
        name = os.path.split(pial_file)[1][len(subject_id) + 1:-4]
        names.append(name)
        cortical_surfaces.create_surface(faces=tri.astype('uint'), vertices=vert, name=name)
    return cortical_surfaces


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


def get_bad_elecs(blockpath):
    bad_channels_file = os.path.join(blockpath, 'Artifacts', 'badChannels.txt')

    # I think bad channels is 1-indexed but I'm not sure
    if os.path.isfile(bad_channels_file) and os.stat(bad_channels_file).st_size:
        dat = pd.read_csv(bad_channels_file, header=None, delimiter=' ', engine='python')
        bad_elecs_inds = dat.values.ravel() - 1
        bad_elecs_inds = bad_elecs_inds[np.isfinite(bad_elecs_inds)]
    else:
        bad_elecs_inds = []

    return bad_elecs_inds


def read_electrodes(elec_metadata_file, load_pos=True):
    """Read metadata for all electrodes

    Parameters
    ----------
    elec_metadata_file: str
    load_pos: bool

    Returns
    -------
    elec_grp_df, coord

    """
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

    anatomy = {'loc': elec_grp_loc, 'type': elec_grp_type,
               'long_name': elec_grp_long_name, 'short_name': elec_grp_short_name,
               'device': elec_grp_device}
    elec_grp_df = pd.DataFrame(anatomy)

    if load_pos:
        n = len(elec_grp_long_name)
        if n < len(elec_grp_xyz_coord):
            coord = elec_grp_xyz_coord[:n]
        elif n == len(elec_grp_xyz_coord):
            coord = elec_grp_xyz_coord
        else:
            coord = elec_grp_xyz_coord
            for i in range(n - len(elec_grp_xyz_coord)):
                coord.append([np.nan, np.nan, np.nan])
    else:
        coord = np.ones((len(elec_grp_long_name), 3)) * np.nan

    return elec_grp_df, coord


def write_electrodes(nwbfile, elec_grp_df, coord, bad_elecs_inds, warped_coord=None):
    ignore_devices = ('NaN', 'Right', 'EKG', 'Na')

    elec_grp_df['bad'] = np.zeros((len(elec_grp_df),), dtype=bool)
    elec_grp_df.loc[bad_elecs_inds, 'bad'] = True

    devices = remove_duplicates(elec_grp_df['device'])
    devices = [x for x in devices if x not in ignore_devices]

    if coord is None:
        coord = np.ones((len(elec_grp_df), 3)) * np.nan

    if warped_coord is not None:
        nwbfile.add_electrode_column('x_warped', 'x warped onto cvs_avg35_inMNI152')
        nwbfile.add_electrode_column('y_warped', 'x warped onto cvs_avg35_inMNI152')
        nwbfile.add_electrode_column('z_warped', 'x warped onto cvs_avg35_inMNI152')

    for device_name in devices:
        device_data = elec_grp_df[elec_grp_df['device'] == device_name]
        elec_counter = 0
        # Create devices
        device = nwbfile.create_device(device_name)

        # Create electrode groups
        electrode_group = nwbfile.create_electrode_group(
            name=device_name + ' electrodes',
            description=device_name,
            location=device_data['type'].iloc[0],
            device=device
        )

        for _, elec_data in device_data.iterrows():
            kwargs = {}
            kwargs.update(x=float(coord[elec_counter, 0]),
                          y=float(coord[elec_counter, 1]),
                          z=float(coord[elec_counter, 2]),
                          imp=np.nan,
                          location=elec_data['loc'],
                          filtering='none',
                          group=electrode_group,
                          bad=elec_data['bad'])
            if warped_coord is not None:
                kwargs.update(x_warped=float(warped_coord[elec_counter, 0]),
                              y_warped=float(warped_coord[elec_counter, 1]),
                              z_warped=float(warped_coord[elec_counter, 2]))
            nwbfile.add_electrode(**kwargs)

            elec_counter += 1


def add_electrodes(nwbfile, elec_metadata_file, bad_elecs_inds, load_warped=True,
                   load_pos=True):
    elec_grp_df, coord = read_electrodes(elec_metadata_file, load_pos=load_pos)

    if load_warped:
        warped_elec_metadata_file = elec_metadata_file[:-4] + '_warped.mat'
        warped_coord = read_electrodes(warped_elec_metadata_file)[1]
    else:
        warped_coord = None

    write_electrodes(nwbfile, elec_grp_df, coord, bad_elecs_inds, warped_coord=warped_coord)


def transpose_iter(aa):
    for i in range(aa.shape[-1]):
        yield aa[..., i].T


def chang2nwb(blockpath, outpath=None, session_start_time=None,
              session_description=None, identifier=None, anin4=False,
              ecog_format='auto', external_subject=True, include_pitch=False, include_intensity=False,
              speakers=True, mic=False, mini=False, hilb=False, verbose=False,
              imaging_path=None, parse_transcript=False, include_cortical_surfaces=True,
              include_electrodes=True, include_ekg=True, subject_image_list=None, rest_period=None,
              load_warped=False, **kwargs):
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
        ({'htk'}, 'mat', 'raw')
    external_subject: bool (optional)
        True: (default) cortical mesh is saved in an external file and a link is
            provided to that file. This is useful if you have multiple sessions for a single subject.
        False: cortical mesh is saved normally
    include_pitch: bool (optional)
        add pitch data. Default: False
    include_intensity: bool (optional)
        add intensity data. Default: False
    speakers: bool (optional)
        Default: False
    mic: bool (optional)
        default: False
    mini: only save data stub. Used for testing
    hilb: bool
        include Hilbert Transform data. Default: False
    verbose: bool (optional)
    imaging_path: str (optional)
        None: use IMAGING_DIR
        'local': use subject_dir/Imaging/
        else: use supplied string
    parse_transcript: str (optional)
    include_cortical_surfaces: bool (optional)
    include_electrodes: bool (optional)
    include_ekg: bool (optional)
    subject_image_list: list (optional)
        List of paths of images to include
    rest_period: None | array-like
    kwargs: dict
        passed to pynwb.NWBFile

    Returns
    -------

    """

    behav_module = None

    basepath, blockname = os.path.split(blockpath)
    subject_id = get_subject_id(blockname)
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
        subj_imaging_path = path.join(IMAGING_PATH, subject_id)
    elif imaging_path == 'local':
        subj_imaging_path = path.join(basepath, 'imaging')
    else:
        subj_imaging_path = os.path.join(imaging_path, subject_id)

    # file paths
    bad_time_file = path.join(blockpath, 'Artifacts', 'badTimeSegments.mat')
    ecog_path = path.join(blockpath, 'RawHTK')
    ecog400_path = path.join(blockpath, 'ecog400', 'ecog.mat')
    elec_metadata_file = path.join(subj_imaging_path, 'elecs', 'TDT_elecs_all.mat')
    mesh_path = path.join(subj_imaging_path, 'Meshes')
    pial_files = glob.glob(path.join(mesh_path, '*pial.mat'))

    # Create the NWB file object
    nwbfile = NWBFile(session_description, identifier,
                      session_start_time, datetime.now().astimezone(),
                      session_id=identifier,
                      institution='University of California, San Francisco',
                      lab='Chang Lab', **kwargs)

    nwbfile.add_electrode_column('bad', 'electrode identified as too noisy')

    bad_elecs_inds = get_bad_elecs(blockpath)

    if include_electrodes:
        add_electrodes(nwbfile, elec_metadata_file, bad_elecs_inds, load_warped=load_warped)
    else:
        device = nwbfile.create_device('256Grid')
        electrode_group = nwbfile.create_electrode_group(
            name='256Grid electrodes', description='auto_group', location='location',
            device=device)

        for elec_counter in range(256):
            bad = elec_counter in bad_elecs_inds
            nwbfile.add_electrode(id=elec_counter+1, x=np.nan, y=np.nan, z=np.nan, imp=np.nan,
                                  location=' ', filtering='none', group=electrode_group, bad=bad)
    ecog_elecs = list(range(len(nwbfile.electrodes)))
    ecog_elecs_region = nwbfile.create_electrode_table_region(ecog_elecs, 'ECoG electrodes on brain')

    # Read electrophysiology data from HTK files and add them to NWB file
    if ecog_format == 'auto':
        ecog_rate, data, ecog_path = auto_ecog(blockpath, ecog_elecs, verbose=False)
    elif ecog_format == 'htk':
        if verbose:
            print('reading htk acquisition...', flush=True)
        ecog_rate, data = readhtks(ecog_path, ecog_elecs)
        data = data.squeeze()
        if verbose:
            print('done', flush=True)

    elif ecog_format == 'mat':
        with File(ecog400_path, 'r') as f:
            data = f['ecogDS']['data'][:, ecog_elecs]
            ecog_rate = f['ecogDS']['sampFreq'][:].ravel()[0]
        ecog_path = ecog400_path

    elif ecog_format == 'raw':
        ecog_path = os.path.join(tdt_data_path, subject_id, blockname, 'raw.mat')
        ecog_rate, data = load_wavs(ecog_path)

    else:
        raise ValueError('unrecognized argument: ecog_format')

    ts_desc = "all Wav data"

    if mini:
        data = data[:2000]

    ecog_ts = ElectricalSeries(name='ElectricalSeries', data=H5DataIO(data, compression='gzip'),
                               electrodes=ecog_elecs_region, rate=ecog_rate, description=ts_desc,
                               conversion=0.001)
    nwbfile.add_acquisition(ecog_ts)

    if include_ekg:
        ekg_elecs = find_ekg_elecs(elec_metadata_file)
        if len(ekg_elecs):
            add_ekg(nwbfile, ecog_path, ekg_elecs)

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
        if fs is not None:
            nwbfile.add_stimulus(TimeSeries('speaker 2', data, 'NA', rate=fs,
                                            description='the second stimulus source'))

    if anin4:
        fs, data = get_analog(blockpath, 4)
        nwbfile.add_acquisition(TimeSeries(anin4, data, 'aux unit', rate=fs,
                                           description="aux analog recording"))

    # Add bad time segments
    if os.path.exists(bad_time_file) and os.stat(bad_time_file).st_size:
        bad_time = sio.loadmat(bad_time_file)['badTimeSegments']
        for row in bad_time:
            nwbfile.add_invalid_time_interval(start_time=row[0], stop_time=row[1],
                                              tags=('ECoG artifact',), timeseries=ecog_ts)

    if rest_period is not None:
        nwbfile.add_epoch_column(name='label', description='label')
        nwbfile.add_epoch(start_time=rest_period[0], stop_time=rest_period[1], label='rest_period')

    if hilb:
        block_hilb_path = os.path.join(hilb_dir, subject_id, blockname, blockname + '_AA.h5')
        file = File(block_hilb_path, 'r')

        data = transpose_iter(file['X'])  # transposes data during iterative write
        filter_center = file['filter_center'][:]
        filter_sigma = file['filter_sigma'][:]

        data = H5DataIO(
            DataChunkIterator(
                tqdm(data, desc='writing hilbert data'),
                buffer_size=400 * 20), compression='gzip')

        decomp_series = DecompositionSeries(
            name='LFPDecompositionSeries',
            description='Gaussian band Hilbert transform',
            data=data, rate=400.,
            source_timeseries=ecog_ts, metric='amplitude')

        for band_mean, band_stdev in zip(filter_center, filter_sigma):
            decomp_series.add_band(band_mean=band_mean, band_stdev=band_stdev)

        hilb_mod = nwbfile.create_processing_module(
            name='ecephys', description='holds hilbert analysis results')
        hilb_mod.add_container(decomp_series)

    if include_cortical_surfaces:
        subject = ECoGSubject(subject_id=subject_id)
        subject.cortical_surfaces = create_cortical_surfaces(pial_files, subject_id)
    else:
        subject = Subject(subject_id=subject_id, species='Homo sapiens')

    if subject_image_list is not None:
        subject = add_images_to_subject(subject, subject_image_list)

    if external_subject:
        subj_fpath = path.join(out_base_path, subject_id + '.nwb')
        if not os.path.isfile(subj_fpath):
            subj_nwbfile = NWBFile(
                session_description=subject_id, identifier=subject_id, subject=subject,
                session_start_time=datetime(1900, 1, 1).astimezone(timezone('UTC')))
            with NWBHDF5IO(subj_fpath, manager=manager, mode='w') as subj_io:
                subj_io.write(subj_nwbfile)
        subj_read_io = NWBHDF5IO(subj_fpath, manager=manager, mode='r')
        subj_nwbfile = subj_read_io.read()
        subject = subj_nwbfile.subject

    nwbfile.subject = subject

    if parse_transcript:
        if parse_transcript == 'CV':
            parseout = parse(blockpath, blockname)
            df = make_df(parseout, 0, subject_id, align_pos=1)
            nwbfile.add_trial_column(
                'cv_transition_time', 'time of CV transition in seconds')
            nwbfile.add_trial_column(
                'speak', 'if True, subject is speaking. If False, subject is listening')
            nwbfile.add_trial_column('condition', 'syllable spoken')
            for _, row in df.iterrows():
                nwbfile.add_trial(
                    start_time=row['start'], stop_time=row['stop'],
                    cv_transition_time=row['align'],
                    speak=row['mode'] == 'speak', condition=row['label'])
        elif parse_transcript == 'singing':
            parseout = parse(blockpath, blockname)
            df = make_df(parseout, 0, subject_id, align_pos=0)
            if not len(df):
                df = pd.DataFrame(parseout)
                df['mode'] = 'speak'

            df = df.loc[df['label'].astype('bool'), :]  # handle empty labels
            nwbfile.add_trial_column(
                'speak', 'if True, subject is speaking. If False, subject is listening')
            nwbfile.add_trial_column('condition', 'syllable spoken')
            for _, row in df.iterrows():
                nwbfile.add_trial(
                    start_time=row['start'], stop_time=row['stop'],
                    speak=row['mode'] == 'speak', condition=row['label'])
        elif parse_transcript == 'emphasis':
            parseout = parse(blockpath, blockname)
            try:
                df = make_df(parseout, 0, subject_id, align_pos=0)
            except:
                df = pd.DataFrame(parseout)
            if not len(df):
                df = pd.DataFrame(parseout)
            df = df.loc[df['label'].astype('bool'), :]  # handle empty labels
            nwbfile.add_trial_column('condition', 'word emphasized')
            nwbfile.add_trial_column(
                'speak', 'if True, subject is speaking. If False, subject is listening')
            for _, row in df.iterrows():
                nwbfile.add_trial(
                    start_time=row['start'], stop_time=row['stop'],
                    speak=True, condition=row['label'])
        elif parse_transcript == 'MOCHA':
            nwbfile = create_transcription(nwbfile, transcript_path, blockname)

    # behavior
    if include_pitch:
        if behav_module is None:
            behav_module = nwbfile.create_processing_module('behavior', 'processing about behavior')
        if os.path.isfile(os.path.join(blockpath, 'pitch_' + blockname + '.mat')):
            fs, data = load_pitch(blockpath)
            pitch_ts = TimeSeries(data=data, rate=fs, unit='Hz', name='pitch',
                                  description='Pitch as extracted from Praat. NaNs mark unvoiced regions.')
            behav_module.add_container(BehavioralTimeSeries(name='pitch', time_series=pitch_ts))
        else:
            print('No pitch file for ' + blockname)

    if include_intensity:
        if behav_module is None:
            behav_module = nwbfile.create_processing_module('behavior', 'processing about behavior')
        if os.path.isfile(os.path.join(blockpath, 'intensity_' + blockname + '.mat')):
            fs, data = load_pitch(blockpath)
            intensity_ts = TimeSeries(data=data, rate=fs, unit='dB', name='intensity',
                                      description='Intensity of speech in dB extracted from Praat.')
            behav_module.add_container(BehavioralTimeSeries(name='intensity', time_series=intensity_ts))
        else:
            print('No intensity file for ' + blockname)

    # Export the NWB file
    with NWBHDF5IO(outpath, manager=manager, mode='w') as io:
        io.write(nwbfile)

    if external_subject:
        subj_read_io.close()

    if hilb:
        file.close()

    # read check
    with NWBHDF5IO(outpath, manager=manager, mode='r') as io:
        io.read()


def gen_external_subject(subject_id, basepath=None, imaging_path=None, outpath=None, subject_image_list=None):

    subject = ECoGSubject(subject_id=subject_id)

    if subject_image_list is not None:
        subject = add_images_to_subject(subject, subject_image_list)

    if imaging_path is None:
        subj_imaging_path = path.join(IMAGING_PATH, subject_id)
    elif imaging_path == 'local':
        subj_imaging_path = path.join(basepath, 'imaging')
    else:
        subj_imaging_path = os.path.join(imaging_path, subject_id)

    mesh_path = path.join(subj_imaging_path, 'Meshes')
    pial_files = glob.glob(path.join(mesh_path, subject_id + '*pial.mat'))

    subject.cortical_surfaces = create_cortical_surfaces(pial_files, subject_id)

    out_base_path = os.path.split(outpath)[0]
    subj_fpath = path.join(out_base_path, subject_id + '.nwb')
    subj_nwbfile = NWBFile(
        session_description=subject_id, identifier=subject_id, subject=subject,
        session_start_time=datetime(1900, 1, 1).astimezone(timezone('UTC')))
    with NWBHDF5IO(subj_fpath, manager=manager, mode='w') as subj_io:
        subj_io.write(subj_nwbfile)


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
