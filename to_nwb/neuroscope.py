"""
Author: Ben Dichter
"""
import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pynwb.ecephys import ElectricalSeries, LFP, SpikeEventSeries, Clustering
from pynwb.behavior import SpatialSeries
from pynwb.misc import AnnotationSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb.form.data_utils import DataChunkIterator
from tqdm import tqdm
from glob import glob


def load_xml(filepath):
    with open(filepath, 'r') as xml_file:
        contents = xml_file.read()
        soup = BeautifulSoup(contents, 'xml')
    return soup


def get_channel_groups(session_path):
    """Get the groups of channels that are recorded on each shank from the xml
    file

    Parameters
    ----------
    session_path: str

    Returns
    -------
    list(list)

    """

    fpath_base, fname = os.path.split(session_path)
    xml_filepath = os.path.join(session_path, fname + '.xml')

    soup = load_xml(xml_filepath)

    channel_groups = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.channelGroups.find_all('group')]

    return channel_groups


def get_shank_channels(session_path):
    """Read the channels on the shanks in Neuroscope xml

    Parameters
    ----------
    session_path: str

    Returns
    -------

    """
    fpath_base, fname = os.path.split(session_path)
    xml_filepath = os.path.join(session_path, fname + '.xml')

    soup = load_xml(xml_filepath)

    shank_channels = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.spikeDetection.channelGroups.find_all('group')]
    return shank_channels


def get_lfp_sampling_rate(session_path):
    """Reads the LFP Sampling Rate from the xml parameter file of the
    Neuroscope format

    Parameters
    ----------
    session_path: str

    Returns
    -------
    fs: float

    """

    session_name = os.path.split(session_path)[1]
    xml_filepath = os.path.join(session_path, session_name + '.xml')

    return float(load_xml(xml_filepath).lfpSamplingRate.string)


def add_position_data(nwbfile, session_path, fs=1250./32.,
                      names=('x0', 'y0', 'x1', 'y1')):
    """Read raw position sensor data from .whl file

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str
    fs: float
        sampling rate
    names: iterable
        names of column headings

    """
    _, session_name = os.path.split(session_path)
    print('warning: time may not be aligned')
    df = pd.read_csv(os.path.join(session_path, session_name + '.whl'),
                     sep='\t', names=names)

    df.index = np.arange(len(df)) / fs
    df.index.name = 'tt (sec)'

    nwbfile.add_acquisition(
        SpatialSeries('position_sensor0',
                      H5DataIO(df[['x0', 'y0']].values, compression='gzip'),
                      'unknown', description='raw sensor data from sensor 0',
                      timestamps=H5DataIO(df.index.values, compression='gzip'),
                      resolution=np.nan))

    nwbfile.add_acquisition(
        SpatialSeries('position sensor1',
                      H5DataIO(df[['x1', 'y1']].values, compression='gzip'),
                      'unknown', description='raw sensor data from sensor 1',
                      timestamps=H5DataIO(df.index.values, compression='gzip'),
                      resolution=np.nan))


def read_spike_times(session_path, shankn, fs=20000.):
    """
    Read .res files

    Parameters
    ----------
    session_path str | path
    shankn: int
        shank number (1-indexed)
    fs: float
        sampling rate. default = 20000.

    Returns
    -------

    """
    _, session_name = os.path.split(session_path)
    timing_file = os.path.join(session_path, session_name + '.res.' + str(shankn))
    timing_df = pd.read_csv(timing_file, names=('time',))

    return timing_df.values.ravel() / fs


def read_spike_clustering(session_path, shankn):
    """
    Read .clu files to get spike cluster assignments for a single shank

    Parameters
    ----------
    session_path: str | path
    shankn: int
        shank number (1-indexed)

    Returns
    -------
    np.ndarray


    """
    session_name = os.path.split(session_path)[1]
    id_file = os.path.join(session_path, session_name + '.clu.' + str(shankn))
    id_df = pd.read_csv(id_file, names=('id',))
    id_df = id_df[1:]  # the first number is the number of clusters

    return id_df.values.ravel()


def get_clusters_single_shank(session_path, shankn, fs=20000.):
    """Read the spike time data for a from the .res and .clu files for a single
    shank. Automatically removes noise and multi-unit.

    Parameters
    ----------
    session_path: str | path
        session path
    shankn: int
        shank number
    fs: float

    Returns
    -------
    df: pd.DataFrame
        has column named 'id' which indicates cluster id and 'time' which
        indicates spike time.

    """

    session_name = os.path.split(session_path)[1]
    spike_times = read_spike_times(session_path, shankn, fs=fs)
    spike_ids = read_spike_clustering(session_path, shankn)
    df = pd.DataFrame({'id': spike_ids, 'time': spike_times})
    noise_inds = ((df.iloc[:, 0] == 0) | (df.iloc[:, 0] == 1)).values.ravel()
    df = df.loc[np.logical_not(noise_inds)].reset_index(drop=True)

    df['id'] -= 2

    return df


def write_clustering(module_cellular, session_path, shanks, fs=20000.):
    for shankn in shanks:
        df = get_clusters_single_shank(session_path, shankn, fs=fs)
        clustering = Clustering(
            name='shank' + str(shankn) + ' clusters',
            description='shank' + str(shankn), num=df['id'].values,
            peak_over_rms=[], times=df['time'].values)

        module_cellular.add_container(clustering)

    return module_cellular


def write_electrode_table(nwbfile, session_path, electrode_positions=None,
                          impedences=None, locations=None, filterings=None):
    """

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str
    electrode_positions: Iterable(Iterable(float))
    impedences: Iterable(float)
    locations: Iterable(str)
    filterings: Iterable(str)

    Returns
    -------

    """
    fpath_base, fname = os.path.split(session_path)

    shank_channels = get_shank_channels(session_path)
    nwbfile.add_electrode_column('shank', '1-indexed shank numbers')
    nwbfile.add_electrode_column('electrode_description',
                                 description='description of electrode description???')

    electrode_counter = 0
    device = nwbfile.create_device('device', fname + '.xml')
    for shankn, channels in enumerate(shank_channels):
        shankn += 1
        electrode_group = nwbfile.create_electrode_group(
            name='shank{}'.format(shankn),
            description='shank{} electrodes'.format(shankn),
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

            nwbfile.add_electrode(float(pos[0]), float(pos[1]), float(pos[2]),
                                  imp=imp, location=location, filtering=filtering,
                                  group=electrode_group, shank=shankn)

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
    lfp_filepath = os.path.join(session_path, fname + '.eeg')

    lfp_fs = get_lfp_sampling_rate(session_path)
    shank_channels = get_shank_channels(session_path)
    all_shank_channels = np.concatenate(shank_channels)

    nelecs = len(nwbfile.electrodes)

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
        name='all_lfp',
        description='lfp signal for all shank electrodes',
        data=data,
        electrodes=all_table_region,
        conversion=np.nan,
        rate=lfp_fs,
        resolution=np.nan)

    nwbfile.add_acquisition(LFP(name='all_lfp', electrical_series=all_lfp_electrical_series))

    return nwbfile


def write_events(nwbfile, session_path, suffixes=None):
    """

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str
    suffixes: Iterable(str), optional
        The 3-letter names for the events to write. If None, detect all in session_path

    Returns
    -------

    nwbfile

    """
    session_name = os.path.split(session_path)[1]

    if suffixes is None:
        evt_files = glob(os.path.join(session_path, session_name) + '.evt.*') + \
                    glob(os.path.join(session_path, session_name) + '.*.evt')
    else:
        evt_files = [os.path.join(session_path, session_name + s)
                     for s in suffixes]
    ann_mod = nwbfile.create_processing_module(
        'annotations', description='evt files')
    for evt_file in evt_files:
        name = evt_file[-3:]
        df = pd.read_csv(evt_file, sep='\t', names=('time', 'desc'))
        timestamps = df.values[:, 0].astype(float) / 1000
        data = df['desc'].values
        annotation_series = AnnotationSeries(
            name=name, data=data, timestamps=timestamps)
        ann_mod.add_container(annotation_series)

    return nwbfile


def write_spike_waveforms(nwbfile, session_path, shankn):
    """

    Parameters
    ----------
    nwbfile: pynwb.NWBFiles
    session_path: str
    shankn: int

    Returns
    -------

    """

    session_name = os.path.split(session_path)[1]
    xml_filepath = os.path.join(session_path, session_name + '.xml')
    soup = load_xml(xml_filepath)
    nsamps = float(soup.spikes.nSamples.string)

    spk_file = os.path.join(session_path, session_name + '.spk.' + str(shankn))
    group_name = 'shank' + str(shankn)

    elec_idx = np.where(np.array(nwbfile.ec_electrodes['group_name']) == group_name)[0]

    nchan = len(elec_idx)

    table_region = nwbfile.create_electrode_table_region(elec_idx, group_name)

    get_shank_channels(xml_filepath)
    spks = np.fromfile(spk_file, dtype=np.int16).reshape(-1, nsamps, nchan)

    spike_times = read_spike_times(session_path, shankn)

    SpikeEventSeries(name='spike_waveforms', data=spks, timestamps=spike_times, electrodes=table_region)

def write_units(nwbfile, session_path):
    nwbfile.add_unit_column('shank', '1-indexed shank number')
    nwbfile.add_unit_column('cluster_id', '0-indexed id of cluster of shank')

    channel_groups = get_channel_groups(session_path)
    for shankn in range(len(channel_groups)):
        df = get_clusters_single_shank(session_path, shankn + 1)
        for cluster_id, idf in df.groupby('id'):
            nwbfile.add_unit(shank=shankn + 1, spike_times=idf['time'].values,
                             cluster_id=cluster_id)

    return nwbfile
