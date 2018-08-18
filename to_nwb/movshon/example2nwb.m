

addpath(genpath('~/dev/NPMK'));
%%
ns_path = '/Users/bendichter/Desktop/Movshon/data/Data_BlackRock_MWorks_forBenDichter/HT_V4_Textures2_200stimoff_180716_001.ns6';
filename = '/Users/bendichter/Desktop/Movshon/data/Data_BlackRock_MWorks_forBenDichter/HT_V4_Textures2_200stimoff_180716_001.nwb';
nev_path = '/Users/bendichter/Desktop/Movshon/data/Data_BlackRock_MWorks_forBenDichter/HT_V4_Textures2_200stimoff_180716_001.nev';
%%
lfp_data = openNSx(ns_path);
%%
spikes_data = openNEV(nev_path);
%%
file = nwbfile( ...
    'source', 'a test source', ...
    'session_description', 'a test NWB File', ...
    'identifier', 'TEST123', ...
    'session_start_time', datestr([1970, 1, 1, 12, 0, 0], 'yyyy-mm-dd HH:MM:SS'), ...
    'file_create_date', datestr([2017, 4, 15, 12, 0, 0], 'yyyy-mm-dd HH:MM:SS'));

%% electrodes

labels = {lfp_data.ElectrodesInfo.Label};

elecs = [];%zeros(length(labels), 1);
for i = 1:length(labels)
    elecs(i) = strcmp(labels{i}(1:4), 'elec');
end

dev = types.core.Device( ...
    'source', ns_path);
file.general_devices.set('dev1', dev);

eg = types.core.ElectrodeGroup('source', ns_path, ...
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

%%
rv = types.untyped.RegionView('/general/extracellular_ephys/electrodes',...
    {[1 sum(elecs)]});
etr = types.core.ElectrodeTableRegion('data', rv);

%% write lfp

es = types.core.ElectricalSeries( ...
    'source', 'a hypothetical source', ...
    'data', lfp_data.Data(logical(elecs),1:1000)', ...
    'electrode_group', types.untyped.SoftLink('/general/extracellular_ephys/elec1'), ...
    'timestamps', (1:10)',...
    'electrodes', etr);

file.acquisition.set('test_eS', es);

%% write spikes

name = 'spikes';

Spikes = spikes_data.Data.Spikes;
spike_loc = ['/acquisition/' name '/spike_times'];

[sorted_elecs, i_sort] = sort(Spikes.Electrode);
% need to convert timestamps to seconds but don't know how yet
sorted_spike_times = single(Spikes.TimeStamp(i_sort))/1000.;
vd = types.core.VectorData('data', sorted_spike_times);

steps = find(diff(sorted_elecs));
vd_ref = types.untyped.RegionView(spike_loc, {[1 steps(1)]});
for i = 1:length(steps)-1
    vd_ref(end+1) = types.untyped.RegionView(spike_loc, ...
        {[steps(i)+1, steps(i+1)]});
end
vd_ref(end+1) = types.untyped.RegionView(spike_loc, ...
    {[steps(i+1), length(Spikes.TimeStamp)]});

vi = types.core.VectorIndex('data', vd_ref);
ei = types.core.ElementIdentifiers('data', int64(unique(Spikes.Electrode)));
ut = types.core.UnitTimes('spike_times', vd, ...
    'spike_times_index', vi, 'unit_ids', ei);

file.acquisition.set(name, ut);


%% write file




nwbExport(file, filename)

%% test read

nwbRead(filename)