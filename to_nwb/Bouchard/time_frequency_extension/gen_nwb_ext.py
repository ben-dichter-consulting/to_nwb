from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec
from pynwb import register_class, load_namespaces, NWBFile, NWBHDF5IO, get_class
from pynwb.form.utils import docval
from pynwb.file import Subject as original_Subject, NWBContainer, MultiContainerInterface


name = 'time_frequency'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

spec = NWBGroupSpec(
    neurodata_type_def='Hilbert',
    neurodata_type_inc='ElectricalSeries',
    quantity='?',
    doc='',
    datasets=[
        NWBDatasetSpec(
            name='data',
            doc='Analytic amplitude of signal',
            dtype='float',
            shape=('null', 'null', 'null'),
            dims=('time', 'channel', 'frequency'),
            attributes=[
                NWBAttributeSpec(
                    name='filter_centers',
                    doc='in Hz',
                    dtype='float'),
                NWBAttributeSpec(
                    name='filter_sigmas',
                    doc='in Hz',
                    dtype='float')
            ]
        )
    ]
)



ns_builder = NWBNamespaceBuilder(name, name)

specs = (spec,)
for spec in specs:
    ns_builder.add_spec(ext_source, spec)
ns_builder.export(ns_path)