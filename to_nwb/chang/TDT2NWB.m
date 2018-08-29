blockpath = '/Users/bendichter/Desktop/Chang/data/TDTBackup/EC125_B22';
elecspath = '/Users/bendichter/Desktop/Chang/data/EC125/Imaging/elecs/TDT_elecs_all.mat';

[~,blockname] = fileparts(blockpath);

%%
tdt = TDTbin2mat(blockpath);

%%
date = datevec([tdt.info.date tdt.info.utcStartTime]);

file = nwbfile( ...
    'source', blockpath, ...
    'session_description', 'a test NWB File', ...
    'identifier', blockname, ...
    'session_start_time', datestr(date, 'yyyy-mm-dd HH:MM:SS'), ...
    'file_create_date', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
%% Electrode Table

load(elecspath);

device_labels = {};
for i = 1:length(anatomy)
    elec_label = anatomy{i,2};
    device_labels{i} = elec_label(1:strfind(elec_label, 'Electrode')-1);
end

udevice_labels = unique(device_labels, 'stable');

variables = {'id', 'x', 'y', 'z', 'imp', 'location', 'filtering', ...
    'description', 'group', 'group_name'};
for i_device = 1:length(udevice_labels)
    device_label = udevice_labels{i_device};
    if ~isempty(device_label) % take care of 'NaN' label
    
        file.general_devices.set(device_label,...
            types.core.Device('source', blockpath));

        file.general_extracellular_ephys.set(device_label,...
            types.core.ElectrodeGroup('source', blockpath, ...
            'description', 'a test ElectrodeGroup', ...
            'location', 'unknown', ...
            'device', types.untyped.SoftLink(['/general/devices/' device_label])));

        ov = types.untyped.ObjectView(['/general/extracellular_ephys/' device_label]);
        
        
        if i_device == 1 && i_elec == 1
            tbl = table(int64(1), x, y, z, NaN, {'location'}, {'filtering'}, ...
                labels(1), ov, {'electrode_group'},...
                'VariableNames', variables);
        end
    end
    
end




%% ECoG

stream_names = fieldnames(tdt.streams);

ecog_stream_names = sort(stream_names(contains(stream_names,'Wav')));

for i = 1:length(ecog_stream_names)
    stream = tdt.streams.(ecog_stream_names{i});
    types.core.ElectricalSeries('source', blockpath,...
        'starting_time',stream.startTime,...
        'starting_time_rate',stream.fs,...
        'data',stream.data',...
        'data_units','unknown. Probably mV?',...
        'electrodes')
end



