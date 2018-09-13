import os
from datetime import datetime

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject

import to_nwb.neuroscope as ns

session_path = '/Users/bendichter/Desktop/Buzsaki/data/buzsakilab.nyumc.org/datasets/McKenzieS/camkii4/20160817'

stub = False

subject_path, session_id = os.path.split(session_path)
subject_id = os.path.split(subject_path)[1]


nwbfile = NWBFile(source=session_path,
                  session_description='session_description',
                  identifier=subject_id + '_' + session_id,
                  session_start_time=datetime.now(),
                  file_create_date=datetime.now(),
                  experimenter='experimenter',
                  session_id=session_id,
                  institution='institution',
                  lab='lab',
                  related_publications='pubs')

nwbfile.subject = Subject(subject_id=subject_id, species='species', source='source')

nwbfile = ns.write_electrode_table(nwbfile, session_path)

nwbfile = ns.write_lfp(nwbfile, session_path, stub=stub)

ut_obj = ns.build_unit_times(session_path)

module_cellular = nwbfile.create_processing_module(
    'cellular', source=session_path, description='holds cellular data')

module_cellular.add_container(ut_obj)

nwbfile = ns.write_events(nwbfile, session_path)

out_fname = session_path
if stub:
    out_fname += '_stub'
out_fname += '.nwb'

with NWBHDF5IO(out_fname, 'w') as io:
    io.write(nwbfile)

with NWBHDF5IO(out_fname, 'r') as io:
    io.read()


