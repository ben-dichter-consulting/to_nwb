import os

from datetime import datetime
import numpy as np
from scipy.io import loadmat
from dateutil.parser import parse as dateparse
import pandas as pd

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.file import Subject, DynamicTable
from pynwb.behavior import SpatialSeries, Position
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.form.data_utils import DataChunkIterator
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from ..extensions.general import CatCellInfo
from tqdm import tqdm


from to_nwb.utils import find_discontinuities
from to_nwb import neuroscope as ns

from ephys_analysis.analysis import filter_lfp, hilbert_lfp


def parse_states(fpath):

    state_map = {'H': 'Home', 'M': 'Maze', 'St': 'LDstim',
                 'O': 'Old open field', 'Oc': 'Old open field w/ curtain',
                 'N': 'New open field', 'Ns': 'New open field w/ LDstim',
                 '5hS': '5 hrs of 1 sec every 5 sec', 'L': 'Large open field',
                 'X': 'Extra large open field',
                 'Nc': 'New open field w/ curtain'}

    subject_path, fname = os.path.split(fpath)
    fpath_base, fname = os.path.split(subject_path)
    subject_id, date_text = fname.split('-')
    session_date = dateparse(date_text, yearfirst=True)
    mouse_num = ''.join(filter(str.isdigit, subject_id))
    exp_sheet_path = os.path.join(subject_path, 'YM' + mouse_num + ' exp_sheet.xlsx')
    df = pd.read_excel(exp_sheet_path, sheet_name=1)
    state_ids = df[df['implanted'] == session_date].values[0, 2:15]

    statepath = os.path.join(fpath, 'EEGlength')
    state_times = pd.read_csv(statepath).values
    states = [state_map[x] for x, _ in zip(state_ids, state_times)]

    return states, state_times


def yuta2nwb(session_path='/Users/bendichter/Desktop/Buzsaki/SenzaiBuzsaki2017/YutaMouse41/YutaMouse41-150903',
             subject_xls=None, stub=False):


    subject_path, session_name = os.path.split(session_path)
    fpath_base = os.path.split(subject_path)[0]
    identifier = session_name
    subject_id, date_text = session_name.split('-')

    if subject_xls is None:
        subject_xls = os.path.join(subject_path, 'YM' + session_name[9:11] + ' exp_sheet.xlsx')

    session_start_time = dateparse(date_text, yearfirst=True)

    df = pd.read_excel(subject_xls)

    subject_data = {}
    for key in ['genotype', 'DOB', 'implantation', 'Probe']:
        subject_data[key] = df.iloc[np.where(df.iloc[:, 0] == key)[0], 1].values[0]

    age = session_start_time - subject_data['DOB']

    subject = Subject(subject_id=subject_id, age=str(age),
                      genotype=subject_data['genotype'],
                      species='mouse', source='source')

    source = session_name
    nwbfile = NWBFile(source=source,
                      session_description='mouse in open exploration and theta maze',
                      identifier=identifier,
                      session_start_time=session_start_time,
                      file_create_date=datetime.now(),
                      experimenter='Yuta Senzai',
                      session_id=session_name,
                      institution='NYU',
                      lab='Buzsaki',
                      subject=subject,
                      related_publications='DOI:10.1016/j.neuron.2016.12.011')

    all_ts = []

    xml_filepath = os.path.join(session_path, session_name + '.xml')

    shank_channels = ns.get_shank_channels(xml_filepath)
    all_shank_channels = np.concatenate(shank_channels)
    lfp_fs = ns.get_lfp_sampling_rate(xml_filepath)

    lfp_channel = 0  # value taken from Yuta's spreadsheet

    print('reading raw position data...', end='', flush=True)
    pos_df = ns.get_position_data(session_path)
    print('done.')

    print('setting up raw position data...', end='', flush=True)
    # raw position sensors file
    pos0 = nwbfile.add_acquisition(
        SpatialSeries('position sensor0',
                      'raw sensor data from sensor 0',
                      H5DataIO(pos_df[['x0', 'y0']].values, compression='gzip'),
                      'unknown',
                      timestamps=H5DataIO(pos_df.index.values, compression='gzip'),
                      resolution=np.nan))
    all_ts.append(pos0)

    pos1 = nwbfile.add_acquisition(
        SpatialSeries('position sensor1',
                      'raw sensor data from sensor 1',
                      H5DataIO(pos_df[['x1', 'y1']].values, compression='gzip'),
                      'unknown',
                      timestamps=H5DataIO(pos_df.index.values, compression='gzip'),
                      resolution=np.nan))
    all_ts.append(pos1)
    print('done.')

    print('setting up electrodes...', end='', flush=True)
    nwbfile.add_electrode_column('electrode_description',
                                 description='description of electrode description???')
    # shank electrodes
    electrode_counter = 0
    for shankn, channels in enumerate(shank_channels):
        device_name = 'shank{}'.format(shankn)
        device = nwbfile.create_device(device_name, session_name + '.xml')
        electrode_group = nwbfile.create_electrode_group(
            name=device_name + '_electrodes',
            source=session_name + '.xml',
            description=device_name,
            device=device,
            location='unknown')
        for channel in channels:
            nwbfile.add_electrode(id=channel, x=np.nan, y=np.nan, z=np.nan,  # position?
                                  imp=np.nan, location='unknown',
                                  filtering='unknown',
                                  electrode_description='electrode {} of shank {}, channel {}'.format(
                                      electrode_counter, shankn, channel),
                                  group=electrode_group)

            if channel == lfp_channel:
                lfp_table_region = nwbfile.create_electrode_table_region(
                    [electrode_counter], 'lfp electrode')

            electrode_counter += 1

    # special electrodes
    device_name = 'special'
    device = nwbfile.create_device(device_name, session_name + '.xml')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        source=session_name + '.xml',
        description=device_name,
        device=device,
        location='unknown')
    special_electrode_dict = {'ch_wait': 79, 'ch_arm': 78, 'ch_solL': 76,
                              'ch_solR': 77, 'ch_dig1': 65, 'ch_dig2': 68,
                              'ch_entL': 72, 'ch_entR': 71, 'ch_SsolL': 73,
                              'ch_SsolR': 70}
    for name, num in special_electrode_dict.items():
        nwbfile.add_electrode(id=num, x=np.nan, y=np.nan, z=np.nan,
                              imp=np.nan, location='unknown',
                              filtering='unknown',
                              electrode_description=name,
                              group=electrode_group)
        nwbfile.create_electrode_table_region([electrode_counter], name)
        electrode_counter += 1

    all_table_region = nwbfile.create_electrode_table_region(
        list(range(electrode_counter)), 'all electrodes')
    print('done.')

    # lfp
    print('reading LFPs...', end='', flush=True)

    if not stub:
        lfp_file = os.path.join(session_path, session_name + '.eeg')
        all_channels = np.fromfile(lfp_file, dtype=np.int16).reshape(-1, 80)
        all_channels_lfp = all_channels[:, all_shank_channels]

        data = DataChunkIterator(tqdm(all_channels_lfp, desc='writing lfp data'),
                                 buffer_size=int(lfp_fs*3600))
        data = H5DataIO(data, compression='gzip')
    else:
        all_channels = np.random.randn(1000, 100)  # use for dev testing for speed
        data = all_channels

    print('done.')

    print('making ElectricalSeries objects for LFP...', end='', flush=True)
    all_lfp_electrical_series = ElectricalSeries(
        'all_lfp',
        'lfp signal for all shank electrodes',
        data,
        all_table_region,
        conversion=np.nan,
        rate=lfp_fs,
        resolution=np.nan)
    all_ts.append(all_lfp_electrical_series)
    nwbfile.add_acquisition(LFP(name='all_lfp', source='source',
                                electrical_series=all_lfp_electrical_series))
    print('done.')

    electrical_series = ElectricalSeries(
        'reference_lfp', 'signal used as the reference lfp',
        H5DataIO(all_channels[:, lfp_channel], compression='gzip'),
        lfp_table_region, conversion=np.nan, rate=lfp_fs, resolution=np.nan)

    nwbfile.add_acquisition(LFP(source='source', name='reference_lfp',
                                electrical_series=electrical_series))
    all_ts.append(electrical_series)

    # create epochs corresponding to experiments/environments for the mouse
    task_types = ['OpenFieldPosition_ExtraLarge', 'OpenFieldPosition_New_Curtain',
                  'OpenFieldPosition_New', 'OpenFieldPosition_Old_Curtain',
                  'OpenFieldPosition_Old', 'OpenFieldPosition_Oldlast', 'EightMazePosition']

    module_behavior = nwbfile.create_processing_module(
        name='behavior', source=source, description=source)
    for label in task_types:

        file = os.path.join(session_path, session_name + '__' + label)
        if os.path.isfile(file):
            print('loading normalized position data for ' + label + '...', end='', flush=True)

            matin = loadmat(file)
            tt = matin['twhl_norm'][:, 0]
            exp_times = find_discontinuities(tt)

            pos_data_norm = matin['twhl_norm'][:, 1:]

            norm_conversion = .65 / (np.max(pos_data_norm[:, 0])
                                     - np.min(pos_data_norm[:, 0]))

            spatial_series_object = SpatialSeries(
                name=label + '_norm_spatial_series', source='position sensor0',
                data=H5DataIO(pos_data_norm, compression='gzip'),
                reference_frame='unknown', conversion=norm_conversion,
                resolution=np.nan,
                timestamps=H5DataIO(tt, compression='gzip'))

            if 'twhl_linearized' in matin:
                pos_data_linearized = matin['twhl_linearized'][:, 1:]

                # each arm is 102 cm. This converts to meters
                lin_conversion = 2.04 / (np.nanmax(pos_data_linearized[:, 1])
                                         - np.nanmin(pos_data_linearized[:, 1]))

                spatial_series_object = [spatial_series_object] + [SpatialSeries(
                    name=label + '_linearized_spatial_series', source='position sensor0',
                    data=H5DataIO(pos_data_linearized, compression='gzip'),
                    reference_frame='unknown', conversion=lin_conversion,
                    resolution=np.nan,
                    timestamps=H5DataIO(tt, compression='gzip'))]

            pos_obj = Position(source=source,
                               spatial_series=spatial_series_object,
                               name=label + '_position')
            module_behavior.add_container(pos_obj)

            for i, window in enumerate(exp_times):
                nwbfile.create_epoch(start_time=window[0], stop_time=window[1],
                                     tags=tuple(), description=label + '_' + str(i),
                                     timeseries=[])
            print('done.')

    # load celltypes
    matin = loadmat(os.path.join(fpath_base, 'DG_all_6__UnitFeatureSummary_add.mat'),
                    struct_as_record=False)['UnitFeatureCell'][0][0]

    # taken from ReadMe
    celltype_dict = {
        0: 'unknown',
        1: 'granule cells (DG) or pyramidal cells (CA3)  (need to use region info. see below.)',
        2: 'mossy cell',
        3: 'narrow waveform cell',
        4: 'optogenetically tagged SST cell',
        5: 'wide waveform cell (narrower, exclude opto tagged SST cell)',
        6: 'wide waveform cell (wider)',
        8: 'positive waveform unit (non-bursty)',
        9: 'positive waveform unit (bursty)',
        10: 'positive negative waveform unit'
    }

    # regions: 3: 'CA3', 4: 'DG'

    this_file = matin.fname == session_name
    celltype_ids = matin.fineCellType.ravel()[this_file]
    region_ids = matin.region.ravel()[this_file]
    unit_ids = matin.unitID.ravel()[this_file]
    shanks = matin.shank.ravel()[this_file] - 1  # change from 1 to 0-indexing

    celltype_names = []
    for celltype_id, region_id in zip(celltype_ids, region_ids):
        if celltype_id == 1:
            if region_id == 3:
                celltype_names.append('pyramidal cell')
            elif region_id == 4:
                celltype_names.append('granule cell')
            else:
                raise Exception('unknown type')
        else:
            celltype_names.append(celltype_dict[celltype_id])

    nwbfile.add_unit_column('cell_type', 'name of cell type')
    nwbfile.add_unit_column('shank', '0-indexed shank number')

    for unit_id, celltype, shank in zip(unit_ids, celltype_names, shanks):
        nwbfile.add_unit({'id': unit_id, 'cell_type': celltype, 'shank': shank})

    ut_obj = ns.build_unit_times(session_path, unit_ids=unit_ids)

    module_cellular = nwbfile.create_processing_module(
        'cellular', source=source, description=source)

    module_cellular.add_container(ut_obj)

    trialdata_path = os.path.join(session_path, session_name + '__EightMazeRun.mat')
    trials_data = loadmat(trialdata_path)['EightMazeRun']

    trialdatainfo_path = os.path.join(fpath_base, 'EightMazeRunInfo.mat')
    trialdatainfo = [x[0] for x in loadmat(trialdatainfo_path)['EightMazeRunInfo'][0]]

    features = trialdatainfo[:7]
    features[:2] = 'start', 'end'
    [nwbfile.add_trial_column(x, 'description') for x in features]

    for trial_data in trials_data:
        nwbfile.add_trial({lab: dat for lab, dat in zip(features, trial_data[:7])})

    mono_syn_fpath = os.path.join(session_path, session_name+'-MonoSynConvClick.mat')

    matin = loadmat(mono_syn_fpath)
    exc = matin['FinalExcMonoSynID']
    inh = matin['FinalInhMonoSynID']

    exc_obj = CatCellInfo(name='excitatory_connections',
                          source=session_name + '-MonoSynConvClick.mat',
                          values=[], cell_index=exc[:, 0] - 1, indices=exc[:, 1] - 1)
    module_cellular.add_container(exc_obj)
    inh_obj = CatCellInfo(name='inhibitory_connections',
                          source=session_name + '-MonoSynConvClick.mat',
                          values=[], cell_index=inh[:, 0] - 1, indices=inh[:, 1] - 1)
    module_cellular.add_container(inh_obj)

    sleep_state_fpath = os.path.join(session_path, session_name+'--StatePeriod.mat')
    matin = loadmat(sleep_state_fpath)['StatePeriod']

    table = DynamicTable(name='states', source='source',
                         description='sleep states of animal')
    table.add_column(name='start', description='start time')
    table.add_column(name='end', description='end time')
    table.add_column(name='state', description='sleep state')

    for name in matin.dtype.names:
        for row in matin[name][0][0]:
            table.add_row({'start': row[0], 'end': row[1], 'state': name})

    module_behavior.add_container(table)

    # compute filtered LFP
    module_lfp = nwbfile.create_processing_module(
        'lfp_mod', source=source, description=source)

    for passband in ('theta', 'gamma'):
        lfp_fft = filter_lfp(all_channels[:, lfp_channel], np.array(lfp_fs), passband=passband)
        lfp_phase, _ = hilbert_lfp(lfp_fft)

        time_series = TimeSeries(name=passband + '_phase',
                                 source='ephys_analysis',
                                 data=lfp_phase,
                                 rate=lfp_fs,
                                 unit='radians')

        module_lfp.add_container(time_series)

    out_fname = session_path + '.nwb'
    if stub:
        out_fname = out_fname[:-4] + '_stub.nwb'
    print('writing NWB file...', end='', flush=True)
    with NWBHDF5IO(out_fname, mode='w') as io:
        io.write(nwbfile)
    print('done.')

    print('testing read...', end='', flush=True)
    # test read
    with NWBHDF5IO(out_fname, mode='r') as io:
        io.read()
    print('done.')
