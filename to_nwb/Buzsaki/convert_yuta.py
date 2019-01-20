from __future__ import print_function

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.parser import parse as dateparse
from ephys_analysis.band_analysis import filter_lfp, hilbert_lfp
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position
from pynwb.file import Subject, TimeIntervals
from pynwb.form.backends.hdf5.h5_utils import H5DataIO
from pynwb.misc import DecompositionSeries
from scipy.io import loadmat

import to_nwb.neuroscope as ns
from to_nwb.utils import find_discontinuities

# value taken from Yuta's spreadsheet
special_electrode_dict = {'ch_wait': 79, 'ch_arm': 78, 'ch_solL': 76,
                          'ch_solR': 77, 'ch_dig1': 65, 'ch_dig2': 68,
                          'ch_entL': 72, 'ch_entR': 71, 'ch_SsolL': 73,
                          'ch_SsolR': 70}


def get_reference_elec(exp_sheet_path, date):
    df1 = pd.read_excel(exp_sheet_path, header=1, sheet_name=1)
    try:
        take = df1['implanted'].values == date
    except:
        import pdb; pdb.set_trace()
    df2 = pd.read_excel(exp_sheet_path, header=3, sheet_name=1)
    out = df2['h'][take[2:]].values[0]

    return out


def add_special_electrodes(nwbfile, session_path):
    """

    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    session_path: str


    """
    session_name = os.path.split(session_path)[1]
    device_name = 'special'
    device = nwbfile.create_device(device_name, session_name + '.xml')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        description=device_name,
        device=device,
        location='unknown')

    electrode_counter = len(nwbfile.electrodes)
    for name, channel in special_electrode_dict.items():
        nwbfile.add_electrode(
            id=channel, x=np.nan, y=np.nan, z=np.nan, imp=np.nan, location='unknown',
            filtering='unknown', electrode_description=name, group=electrode_group, shank=-1)
        nwbfile.create_electrode_table_region([electrode_counter], name)
        electrode_counter += 1


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
             subject_xls=None, stub=True):

    subject_path, session_name = os.path.split(session_path)
    fpath_base = os.path.split(subject_path)[0]
    identifier = session_name
    subject_id, date_text = session_name.split('-')

    if subject_xls is None:
        subject_xls = os.path.join(subject_path, 'YM' + session_name[9:11] + ' exp_sheet.xlsx')

    session_start_time = dateparse(date_text, yearfirst=True)

    lfp_channel = get_reference_elec(subject_xls, session_start_time)

    df = pd.read_excel(subject_xls)

    subject_data = {}
    for key in ['genotype', 'DOB', 'implantation', 'Probe']:
        subject_data[key] = df.iloc[np.where(df.iloc[:, 0] == key)[0], 1].values[0]

    age = session_start_time - subject_data['DOB']

    subject = Subject(subject_id=subject_id, age=str(age),
                      genotype=subject_data['genotype'],
                      species='mouse')

    nwbfile = NWBFile(session_description='mouse in open exploration and theta maze',
                      identifier=identifier,
                      session_start_time=session_start_time.astimezone(),
                      file_create_date=datetime.now().astimezone(),
                      experimenter='Yuta Senzai',
                      session_id=session_name,
                      institution='NYU',
                      lab='Buzsaki',
                      subject=subject,
                      related_publications='DOI:10.1016/j.neuron.2016.12.011')

    print('reading and writing raw position data...', end='', flush=True)
    ns.add_position_data(nwbfile, session_path)

    print('setting up electrodes...', end='', flush=True)
    ns.write_electrode_table(nwbfile, session_path)

    add_special_electrodes(nwbfile, session_path)

    print('reading LFPs...', end='', flush=True)
    lfp_fs, all_channels_data = ns.read_lfp(session_path, stub=stub)
    shank_channels = ns.get_shank_channels(session_path)
    all_shank_channels = np.concatenate(shank_channels)
    lfp_data = all_channels_data[:, all_shank_channels]
    print('writing LFPs...', flush=True)
    ns.write_lfp(nwbfile, lfp_data, lfp_fs, name='all_lfp',
                 description='lfp signal for all shank electrodes')

    reference_lfp_data = all_channels_data[:, lfp_channel]

    lfp_index = np.where(all_shank_channels == lfp_channel)[0][0]

    reference_lfp_ts = ns.write_lfp(nwbfile, reference_lfp_data, lfp_fs, name='reference_lfp',
                                    description='lfp signal for reference electrode', electrode_inds=[lfp_index])

    # create epochs corresponding to experiments/environments for the mouse
    task_types = ['OpenFieldPosition_ExtraLarge', 'OpenFieldPosition_New_Curtain',
                  'OpenFieldPosition_New', 'OpenFieldPosition_Old_Curtain',
                  'OpenFieldPosition_Old', 'OpenFieldPosition_Oldlast', 'EightMazePosition']

    module_behavior = nwbfile.create_processing_module(name='behavior', description='description')
    nwbfile.add_epoch_column('label', 'name of epoch')
    for label in task_types:

        file = os.path.join(session_path, session_name + '__' + label + '.mat')
        if os.path.isfile(file):
            print('loading normalized position for ' + label + '...', end='', flush=True)

            matin = loadmat(file)
            tt = matin['twhl_norm'][:, 0]
            exp_times = find_discontinuities(tt)

            pos_data_norm = matin['twhl_norm'][:, 1:]

            norm_conversion = .65 / (np.max(pos_data_norm[:, 0])
                                     - np.min(pos_data_norm[:, 0]))

            spatial_series_object = SpatialSeries(
                name=label + '_norm_spatial_series',
                data=H5DataIO(pos_data_norm, compression='gzip'),
                reference_frame='unknown', conversion=norm_conversion,
                resolution=np.nan,
                timestamps=H5DataIO(tt, compression='gzip'))

            if 'twhl_linearized' in matin:
                print('loading linearized position...', end='', flush=True)
                pos_data_linearized = matin['twhl_linearized'][:, 1:]

                # each arm is 102 cm. This converts to meters
                lin_conversion = 2.04 / (np.nanmax(pos_data_linearized[:, 1])
                                         - np.nanmin(pos_data_linearized[:, 1]))

                spatial_series_object = [spatial_series_object] + [SpatialSeries(
                    name=label + '_linearized_spatial_series',
                    data=H5DataIO(pos_data_linearized, compression='gzip'),
                    reference_frame='unknown', conversion=lin_conversion,
                    resolution=np.nan,
                    timestamps=H5DataIO(tt, compression='gzip'))]

            pos_obj = Position(name=label + '_position',
                               spatial_series=spatial_series_object)
            module_behavior.add_container(pos_obj)
            for i, window in enumerate(exp_times):
                nwbfile.add_epoch(start_time=window[0], stop_time=window[1],
                                  label=label + '_' + str(i))
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
    global_ids = matin.unitID.ravel()[this_file]
    shank_ids = matin.unitIDshank.ravel()[this_file] - 2  # 0 and 1 are noised
    shanks = matin.shank.ravel()[this_file]

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
    nwbfile.add_unit_column('shank', '1-indexed shank number')
    nwbfile.add_unit_column('global_id', 'global id for cell for entire experiment')

    for unit_id, celltype, shank, shank_id in zip(global_ids, celltype_names, shanks, shank_ids):
        df = ns.get_clusters_single_shank(session_path, shank)
        spike_times = df['time'][df['id'] == shank_id].values
        nwbfile.add_unit(global_id=unit_id, cell_type=celltype, shank=shank,
                         spike_times=spike_times)

    trialdata_path = os.path.join(session_path, session_name + '__EightMazeRun.mat')
    trials_data = loadmat(trialdata_path)['EightMazeRun']

    trialdatainfo_path = os.path.join(fpath_base, 'EightMazeRunInfo.mat')
    trialdatainfo = [x[0] for x in loadmat(trialdatainfo_path)['EightMazeRunInfo'][0]]

    features = trialdatainfo[:7]
    features[:2] = 'start_time', 'stop_time',
    [nwbfile.add_trial_column(x, 'description') for x in features[4:] + ['condition']]

    for trial_data in trials_data:
        if trial_data[3]:
            cond = 'run_left'
        else:
            cond = 'run_right'
        nwbfile.add_trial(start_time=trial_data[0], stop_time=trial_data[1], condition=cond,
                          error_run=trial_data[4], stim_run=trial_data[5], both_visit=trial_data[6])
    """
    mono_syn_fpath = os.path.join(session_path, session_name+'-MonoSynConvClick.mat')

    matin = loadmat(mono_syn_fpath)
    exc = matin['FinalExcMonoSynID']
    inh = matin['FinalInhMonoSynID']

    #exc_obj = CatCellInfo(name='excitatory_connections',
    #                      indices_values=[], cell_index=exc[:, 0] - 1, indices=exc[:, 1] - 1)
    #module_cellular.add_container(exc_obj)
    #inh_obj = CatCellInfo(name='inhibitory_connections',
    #                      indices_values=[], cell_index=inh[:, 0] - 1, indices=inh[:, 1] - 1)
    #module_cellular.add_container(inh_obj)
    """

    sleep_state_fpath = os.path.join(session_path, session_name+'--StatePeriod.mat')
    matin = loadmat(sleep_state_fpath)['StatePeriod']

    table = TimeIntervals(name='states', description='sleep states of animal')
    table.add_column(name='label', description='sleep state')

    for name in matin.dtype.names:
        for row in matin[name][0][0]:
            table.add_row(start_time=row[0], stop_time=row[1], label=name)

    module_behavior.add_container(table)

    # compute filtered LFP
    print('filtering LFP...', end='', flush=True)
    all_lfp_phases = []
    for passband in ('theta', 'gamma'):
        lfp_fft = filter_lfp(reference_lfp_data, lfp_fs, passband=passband)
        lfp_phase, _ = hilbert_lfp(lfp_fft)
        all_lfp_phases.append(lfp_phase[:, np.newaxis])
    data = np.dstack(all_lfp_phases)
    print('done.', flush=True)

    decomp_series = DecompositionSeries(name='LFPSpectralAnalysis',
                                        description='Theta and Gamma phase for reference LFP',
                                        data=data, rate=lfp_fs,
                                        source_timeseries=reference_lfp_ts,
                                        metric='phase', unit='radians')
    decomp_series.add_band(band_name='theta', band_limits=(4, 10))
    decomp_series.add_band(band_name='gamma', band_limits=(30, 80))

    nwbfile.modules['ecephys'].add_data_interface(decomp_series)

    if stub:
        out_fname = session_path + '_stub.nwb'
    else:
        out_fname = session_path + '.nwb'

    print('writing NWB file...', end='', flush=True)
    with NWBHDF5IO(out_fname, mode='w') as io:
        io.write(nwbfile)
    print('done.')

    print('testing read...', end='', flush=True)
    # test read
    with NWBHDF5IO(out_fname, mode='r') as io:
        io.read()
    print('done.')


def main(argv):
    yuta2nwb()


if __name__ == "__main__":
    main(sys.argv[1:])
