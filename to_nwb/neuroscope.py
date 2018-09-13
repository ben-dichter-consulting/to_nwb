import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.misc import AnnotationSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb.form.data_utils import DataChunkIterator
from pynwb.misc import UnitTimes
from tqdm import tqdm
from glob import glob


def load_xml(filepath):
    with open(filepath, 'r') as xml_file:
        contents = xml_file.read()
        soup = BeautifulSoup(contents, 'xml')
    return soup


def get_channel_groups(xml_filepath):
    """Get the groups of channels that are recorded on each shank from the xml
    file

    Parameters
    ----------
    xml_filepath: str

    Returns
    -------
    list(list)

    """

    soup = load_xml(xml_filepath)

    channel_groups = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.channelGroups.find_all('group')]

    return channel_groups


def get_shank_channels(xml_filepath):
    """Read the channels on the shanks in Neuroscope xml

    Parameters
    ----------
    xml_filepath: str

    Returns
    -------

    """
    soup = load_xml(xml_filepath)

    shank_channels = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.spikeDetection.channelGroups.find_all('group')]
    return shank_channels


def get_lfp_sampling_rate(xml_filepath):
    """Reads the LFP Sampling Rate from the xml parameter file of the
    Neuroscope format

    Parameters
    ----------
    xml_filepath: str

    Returns
    -------
    fs: float

    """
    return float(load_xml(xml_filepath).lfpSamplingRate.string)


def get_position_data(session_path, fs=1250./32.,
                      names=('x0', 'y0', 'x1', 'y1')):
    """Read raw position sensor data from .whl file

    Parameters
    ----------
    session_path: str
    fs: float
    names: iterable
        names of column headings

    Returns
    -------
    df: pandas.DataFrame
    """
    _, session_name = os.path.split(session_path)
    print('warning: time may not be aligned')
    df = pd.read_csv(os.path.join(session_path, session_name + '.whl'),
                     sep='\t', names=names)

    df.index = np.arange(len(df)) / fs
    df.index.name = 'tt (sec)'

    return df


def get_clusters_single_shank(session_path, shankn, fs=20000):
    """Read the spike time data for a from the .res and .clu files for a single
    shank. Automatically removes noise and multi-unit.

    Parameters
    ----------
    session_path: str | path
        session path
    shankn: int
        shank number

    Returns
    -------
    df: pd.DataFrame
        has column named 'id' which indicates cluster id and 'time' which
        indicates spike time.

    """

    _, session_name = os.path.split(session_path)
    timing_file = os.path.join(session_path, session_name + '.res.' + str(shankn))
    id_file = os.path.join(session_path, session_name + '.clu.' + str(shankn))

    timing_df = pd.read_csv(timing_file, names=('time',))
    id_df = pd.read_csv(id_file, names=('id',))
    id_df = id_df[1:]  # the first number is the number of clusters
    noise_inds = ((id_df == 0) | (id_df == 1)).values.ravel()
    df = id_df.join(timing_df)
    df = df.loc[np.logical_not(noise_inds)].reset_index(drop=True)
    df['time'] = df['time'] / fs

    df['id'] -= 2

    return df


def build_unit_times(session_path, shanks=None, name='UnitTimes', source=None,
                     unit_ids=None):
    """

    Parameters
    ----------
    session_path: str
    shanks: None | list(ints)
        shank numbers to process. If None, use 1:8
    name: str
    source: str
    unit_ids: array-like if ints, optional
        If not provided, count from 0

    Returns
    -------

    """

    if shanks is None:
        shanks = range(1, 9)

    if source is None:
        source = session_path

    ut = UnitTimes(name=name, source=source)

    cell_counter = 0
    for shank_num in shanks:
        df = get_clusters_single_shank(session_path, shank_num)
        for cluster_num, idf, in df.groupby('id'):
            if unit_ids is not None:
                unit_id = unit_ids[cell_counter]
            else:
                unit_id = cell_counter
            ut.add_spike_times(int(unit_id), list(idf['time']))
            cell_counter += 1

    return ut


def write_electrode_table(nwbfile, session_path, electrode_positions=None,
                          impedences=None, locations=None, filterings=None):
    """

    Parameters
    ----------
    nwbfile; pynwb.NWBFile
    session_path: str
    electrode_positions: Iterable(Iterable(float))
    impedences: Iterable(float)
    locations: Iterable(str)
    filterings: Iterable(str)

    Returns
    -------

    """
    fpath_base, fname = os.path.split(session_path)
    xml_filepath = os.path.join(session_path, fname + '.xml')

    shank_channels = get_shank_channels(xml_filepath)

    electrode_counter = 0
    for shankn, channels in enumerate(shank_channels):
        device_name = 'shank{}'.format(shankn)
        device = nwbfile.create_device(device_name, fname + '.xml')
        electrode_group = nwbfile.create_electrode_group(
            name=device_name + '_electrodes',
            source=fname + '.xml',
            description=device_name,
            device=device,
            location='unknown')
        for channel in channels:
            if electrode_positions is not None:
                pos = electrode_positions[channel]
            else:
                pos = (np.nan, np.nan, np.nan)

            if impedences is None:
                imp = np.nan
            else:
                imp = impedences[channel]

            if locations is None:
                location = 'unknown'
            else:
                location = locations[channel]

            if filterings is None:
                filtering = 'unknown'
            else:
                filtering = filterings[channel]

            nwbfile.add_electrode(channel, pos[0], pos[1], pos[2], imp=imp,
                                  location=location, filtering=filtering,
                                  description='electrode {} of shank {}, channel {}'.format(
                                      electrode_counter, shankn, channel),
                                  group=electrode_group)

            electrode_counter += 1

    return nwbfile


def write_lfp(nwbfile, session_path, stub=False):
    """
    Add LFP from neuroscope to NWB.

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str
    stub: bool, optional
        Default is False. If True, don't read LFP, but instead add a small
        amount of placeholder data. This is useful for rapidly checking new
        features without the time-intensive data read step.

    Returns
    -------

    nwbfile

    """
    fpath_base, fname = os.path.split(session_path)
    xml_filepath = os.path.join(session_path, fname + '.xml')
    lfp_filepath = os.path.join(session_path, fname + '.eeg')

    lfp_fs = get_lfp_sampling_rate(xml_filepath)
    shank_channels = get_shank_channels(xml_filepath)
    all_shank_channels = np.concatenate(shank_channels)

    nelecs = len(nwbfile.ec_electrodes)

    all_table_region = nwbfile.create_electrode_table_region(
        list(range(nelecs)), 'all electrodes')

    if stub:
        data = np.random.randn(1000, 100)  # use for dev testing for speed
    else:
        all_channels = np.fromfile(lfp_filepath, dtype=np.int16).reshape(-1, nelecs)
        all_channels_lfp = all_channels[:, all_shank_channels]

        data = DataChunkIterator(tqdm(all_channels_lfp, desc='writing lfp data'),
                                 buffer_size=int(lfp_fs * 3600))
        data = H5DataIO(data, compression='gzip')

    all_lfp_electrical_series = ElectricalSeries(
        'all_lfp',
        'lfp signal for all shank electrodes',
        data,
        all_table_region,
        conversion=np.nan,
        rate=lfp_fs,
        resolution=np.nan)

    nwbfile.add_acquisition(LFP(name='all_lfp', source='source',
                                electrical_series=all_lfp_electrical_series))

    return nwbfile


def write_events(nwbfile, session_path, suffixes=None):
    """

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str
    suffixes: Iterable(str), optional

    Returns
    -------

    nwbfile

    """
    _, session_name = os.path.split(session_path)

    if suffixes is None:
        evt_files = glob(os.path.join(session_path, session_name) + '.evt.*') + \
                    glob(os.path.join(session_path, session_name) + '.*.evt')
    else:
        evt_files = [os.path.join(session_path, session_name + s)
                     for s in suffixes]
    ann_mod = nwbfile.create_processing_module(
        'annotations', source=session_path, description='evt files')
    for evt_file in evt_files:
        name = evt_file[-3:]
        df = pd.read_csv(evt_file, sep='\t', names=('time', 'desc'))
        timestamps = df.values[:, 0].astype(float) / 1000
        data = df['desc'].values
        annotation_series = AnnotationSeries(
            name=name, source=evt_file, data=data, timestamps=timestamps)
        ann_mod.add_container(annotation_series)

    return nwbfile
