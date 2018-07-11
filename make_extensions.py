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
                       name='cell_index', dtype='int', quantity='?'),
        NWBDatasetSpec(name='indices',
                       doc='list of indices for values',
                       shape=(None, 1), dtype='int',
                       attributes=[
                           NWBAttributeSpec(name='values', dtype='text',
                                            doc='values that the indices are indexing',
                                            shape=(None, 1))])],
    neurodata_type_inc='NWBDataInterface')

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
ns_builder.add_spec(ext_source, cat_cell_info)
ns_builder.export(ns_path)
