from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec
from pynwb import register_class, load_namespaces, NWBFile, NWBHDF5IO, get_class
from pynwb.form.utils import docval
from pynwb.file import Subject as original_Subject, NWBContainer, MultiContainerInterface


name = 'time_frequency'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

spec = NWBGroupSpec(
    neurodata_type_def='HilbertSeries',
    neurodata_type_inc='ElectricalSeries',
    quantity='?',
    doc='output of hilbert transform',
    attributes=[
        NWBAttributeSpec(name='unit',
                         doc='unit',
                         dtype='text',
                         value='no units'),
        NWBAttributeSpec(name='help',
                         doc='help',
                         dtype='text',
                         value='ENTER HELP INFO HERE')],
    datasets=[
        NWBDatasetSpec(name='filter_centers',
                       doc='in Hz',
                       dtype='float',
                       shape=('null', 'null')),
        NWBDatasetSpec(name='filter_sigmas',
                       doc='in Hz',
                       dtype='float',
                       shape=('null', 'null')),
        NWBDatasetSpec(
            name='data',
            doc='Analytic amplitude of signal',
            dtype='float',
            shape=('null', 'null', 'null'),
            dims=('time', 'channel', 'frequency'),
            quantity='?'),
        NWBDatasetSpec(
            name='real_data',
            doc='The real component of the complex result of the hilbert transform',
            dtype='float',
            shape=('null', 'null', 'null'),
            dims=('time', 'channel', 'frequency'),
            quantity='?'),
        NWBDatasetSpec(
            name='imaginary_data',
            doc='The imaginary component of the complex result of the hilbert transform',
            dtype='float',
            shape=('null', 'null', 'null'),
            dims=('time', 'channel', 'frequency'),
            quantity='?'),
        NWBDatasetSpec(
            name='phase_data',
            doc='The phase of the complex result of the hilbert transform',
            dtype='float',
            shape=('null', 'null', 'null'),
            dims=('time', 'channel', 'frequency'),
            quantity='?')
    ]
)


ns_builder = NWBNamespaceBuilder(name, name)

specs = (spec,)
for spec in specs:
    ns_builder.add_spec(ext_source, spec)
ns_builder.export(ns_path)

