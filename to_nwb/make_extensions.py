from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, \
    NWBAttributeSpec

name = 'general'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'

values = NWBAttributeSpec(name='values',
                          dtype='text',
                          doc='values that the indices are indexing',
                          shape=(None, 1))

cat_cell_info = NWBGroupSpec(
    neurodata_type_def='CatCellInfo',
    doc='Categorical Cell Info',
    datasets=[
        NWBDatasetSpec(doc='global id for neuron',
                       shape=(None, 1),
                       name='cell_index', dtype='int', quantity='?'),
        NWBDatasetSpec(name='indices',
                       doc='list of indices for values',
                       shape=(None, 1), dtype='int',
                       attributes=[values])],
    neurodata_type_inc='NWBDataInterface')

cat_timeseries = NWBGroupSpec(
    neurodata_type_def='CatTimeSeries',
    neurodata_type_inc='TimeSeries',
    doc='Categorical data through time',
    datasets=[NWBDatasetSpec(name='data',
                             shape=(None, 1), dtype='int',
                             doc='timeseries of indicies for values',
                             attributes=[values])])

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
for spec in (cat_cell_info, cat_timeseries):
    ns_builder.add_spec(ext_source, spec)
ns_builder.export(ns_path)
