function file = AddNSFile(file, ns, starting_time)
% ns can be filepath or data structure

if ~exist('starting_time','var') || isempty(starting_time)
    starting_time = 0.0;
end

source = 'NS file';
if ischar(ns)
    lfp_data = openNSx(ns);
else
    lfp_data = ns;
end

labels = {lfp_data.ElectrodesInfo.Label};

elecs = [];%zeros(length(labels), 1);
for i = 1:length(labels)
    elecs(i) = strcmp(labels{i}(1:4), 'elec');
end

%% Electrode Table

dev = types.core.Device( ...
    'source', 'source');
file.general_devices.set('dev1', dev);

eg = types.core.ElectrodeGroup('source', source, ...
    'description', 'a test ElectrodeGroup', ...
    'location', 'unknown', ...
    'device', types.untyped.SoftLink('/general/devices/dev1'));
file.general_extracellular_ephys.set('electrode_group', eg);
ov = types.untyped.ObjectView('/general/extracellular_ephys/electrode_group');

variables = {'id', 'x', 'y', 'z', 'imp', 'location', 'filtering', ...
    'description', 'group', 'group_name'};
tbl = table(int64(1), NaN, NaN, NaN, NaN, {'location'}, {'filtering'}, ...
    labels(1), ov, {'electrode_group'},...
    'VariableNames', variables);
for i = 2:sum(elecs)
    tbl = [tbl; {int64(i), NaN, NaN, NaN, NaN, 'location', 'filtering', ...
        labels(i), ov, 'electrode_group'}];
end

et = types.core.ElectrodeTable('data', tbl);

file.general_extracellular_ephys.set('electrodes', et);

%% Electrode Table Region

rv = types.untyped.RegionView('/general/extracellular_ephys/electrodes',...
    {[1 sum(elecs)]});

etr = types.core.ElectrodeTableRegion('data', rv);

%% write LFP
elec_info = lfp_data.ElectrodesInfo(find(elecs,1));

[unit, conversion] = blackrock.get_channel_info(elec_info);

es = types.core.ElectricalSeries( ...
    'source', source, ...
    'data', single(lfp_data.Data(logical(elecs),1:100))', ...
    'electrode_group', types.untyped.SoftLink('/general/extracellular_ephys/elec1'), ...
    'starting_time_rate', lfp_data.MetaTags.TimeRes,...
    'starting_time', starting_time,...
    'electrodes', etr,...
    'data_unit',unit,...
    'data_conversion',conversion);

file.acquisition.set('lfp', es);

%% write other acquisition

non_elecs = find(1-elecs);



for i = 1:length(non_elecs)

    channel = non_elecs(i);
    
    label = labels{channel};
    label = label(logical(label));
    
    elec_info = lfp_data.ElectrodesInfo(channel);
    
    [unit, conversion] = blackrock.get_channel_info(elec_info);

    ts = types.core.TimeSeries( ...
        'source', source, ...
        'data', single(lfp_data.Data(channel,:))', ...
        'starting_time_rate', lfp_data.MetaTags.TimeRes,...
        'starting_time', starting_time,...
        'data_unit',unit,...
        'data_conversion',conversion);
    
    file.acquisition.set(label, ts);
    
end