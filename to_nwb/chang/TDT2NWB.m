blockpath = '/Users/bendichter/Desktop/Chang/data/TDTBackup/EC125_B22';
elecspath = '/Users/bendichter/Desktop/Chang/data/EC125/Imaging/elecs/TDT_elecs_all.mat';

[basepath, blockname] = fileparts(blockpath);

nwb_path = fullfile(basepath, [blockname '.nwb']);

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

elecs = load(elecspath, 'anatomy', 'elecmatrix');
location = {elecs.anatomy{:,4}};
x = elecs.elecmatrix(:,1);
y = elecs.elecmatrix(:,2);
z = elecs.elecmatrix(:,3);
label = {elecs.anatomy{:,2}};


device_labels = {};
for i = 1:length(elecs.anatomy)
    this_label = label{i};
    device_labels{i} = this_label(1:strfind(this_label, 'Electrode')-1);
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
        
        elec_nums = find(strcmp(device_labels, device_label));
        for i_elec = 1:length(elec_nums)
            elec_num = elec_nums(i_elec);
            if i_device == 1 && i_elec == 1
                tbl = table(int64(1), x(1), y(1), z(1), NaN, location(1), {'filtering'}, ...
                    label(1), ov, {'electrode_group'},...
                    'VariableNames', variables);
            else
                tbl = [tbl; {int64(elec_num), x(elec_num), y(elec_num), z(elec_num), NaN,...
                    location{elec_num}, 'filtering', label{elec_num}, ov, 'electrode_group'}];
            end
        end
        
    end
end

et = types.core.ElectrodeTable('data', tbl);
file.general_extracellular_ephys.set('electrodes', et);

%% Electrode Table Region

rv = types.untyped.RegionView('/general/extracellular_ephys/electrodes',...
    {[1 length(x)]});

etr = types.core.ElectrodeTableRegion('data', rv);

%% ECoG

stream_names = fieldnames(tdt.streams);

ecog_stream_names = sort(stream_names(contains(stream_names,'Wav')));

Data = [];
for i = 1:length(ecog_stream_names)
    stream = tdt.streams.(ecog_stream_names{i});
    Data = [Data, stream.data'];
end
Data = Data(:,~strcmp(device_labels,''));

es = types.core.ElectricalSeries('source', blockpath,...
    'starting_time',stream.startTime,...
    'starting_time_rate',stream.fs,...
    'data',Data',...
    'electrodes', etr,...
    'data_unit','V');

file.acquisition.set('lfp', es);


%% ANIN

stream = tdt.streams.ANIN;

labels = {'microphone', 'speaker1', 'speaker2', 'anin4'};

for i = 1:length(labels)
    ts = types.core.TimeSeries('source', blockpath,...
        'starting_time',stream.startTime,...
        'starting_time_rate',stream.fs,...
        'data',stream.data(i,:)',...
        'data_unit','V?');
    file.acquisition.set(labels{i}, ts);
end



%% write file
nwbExport(file, nwb_path)

%% test read
nwb_read = nwbRead(nwb_path);




