from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, \
    NWBAttributeSpec

name = 'general'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'

cat_cell_info = NWBGroupSpec(
    neurodata_type_def='CatCellInfo',
    doc='Categorical Cell Info',
    datasets=[
        NWBDatasetSpec(doc='global id for neuron',
                       shape=(None, 1),
                       name='cell_index', dtype='int'),
        NWBDatasetSpec(name='values',
                       doc='list of unique values',
                       attributes=[NWBAttributeSpec(
                           name='indices',
                           doc='indices into values for each gid in order',
                           dtype='int',
                           shape=(None,))],
                       shape=(None, 1), dtype='text')],
    neurodata_type_inc='NWBDataInterface')

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
ns_builder.add_spec(ext_source, cat_cell_info)
ns_builder.export(ns_path)
