from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec
from pynwb import register_class, load_namespaces, NWBFile, NWBHDF5IO, get_class
from pynwb.form.utils import docval
from pynwb.file import Subject as original_Subject, NWBContainer, MultiContainerInterface


name = 'template [CHANGE TO NAME]'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

spec = NWBGroupSpec(
    neurodata_type_def='',
    neurodata_type_inc='',
    quantity='?',
    doc='',
    attributes=[
        NWBAttributeSpec(name='',
                         doc='',
                         dtype='',
                         required=False),
        NWBAttributeSpec(name='help',
                         doc='help',
                         dtype='text',
                         value='ENTER HELP INFO HERE')
        ],
    datasets=[
        NWBDatasetSpec(name='',
                       doc='',
                       dtype='',
                       shape=())
    ]
)

ns_builder = NWBNamespaceBuilder(name, name)

specs = (spec,)
for spec in specs:
    ns_builder.add_spec(ext_source, spec)
ns_builder.export(ns_path)

