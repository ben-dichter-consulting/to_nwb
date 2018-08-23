from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec
from pynwb import register_class, load_namespaces, NWBFile, NWBHDF5IO
from pynwb.form.utils import docval
from pynwb.file import Subject, NWBContainer, MultiContainerInterface, NWBDataInterface
from pynwb.device import Device

import re


name = 'buzsaki_meta'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

virus_injection = NWBGroupSpec(
    neurodata_type_inc='NWBDataInterface',
    neurodata_type_def='VirusInjection', quantity='*',
    doc='notes about surgery that includes virus injection',
    datasets=[NWBDatasetSpec(name='coordinates', doc='(AP, ML, DV) of virus injection',
                             dtype='float', shape=(3,))],
    attributes=[
        NWBAttributeSpec(name='virus', doc='type of virus', dtype='text'),
        NWBAttributeSpec(name='volume', doc='volume of injecting in nL', dtype='float'),
        NWBAttributeSpec(name='rate', doc='rate of injection (nL/s)',
                         dtype='float', required=False),
        NWBAttributeSpec(name='scheme', doc='scheme of injection', dtype='text', required=False),
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Information about a virus injection')])

virus_injections = NWBGroupSpec(
    neurodata_type_def='VirusInjections',
    neurodata_type_inc='NWBDataInterface',
    name='virus_injections',
    doc='stores virus injections', quantity='?',
    groups=[virus_injection],
    attributes=[
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Container for virus injections')
    ])

surgery = NWBGroupSpec(
    neurodata_type_def='Surgery', doc='information about a specific surgery', quantity='+',
    neurodata_type_inc='NWBDataInterface',
    datasets=[NWBDatasetSpec(name='devices', quantity='?', doc='links to implanted/explanted devices',
                             dtype=RefSpec('Device', 'object'))],
    groups=[virus_injections],
    attributes=[
        NWBAttributeSpec(name='start_datetime', doc='datetime in ISO 8601', dtype='text', required=False),
        NWBAttributeSpec(name='end_datetime', doc='datetime in ISO 8601', dtype='text', required=False),
        NWBAttributeSpec(name='weight', required=False, dtype='text',
                         doc='Weight at time of experiment, at time of surgery and at other '
                             'important times'),
        NWBAttributeSpec(name='notes', doc='notes and complications', dtype='text', required=False),
        NWBAttributeSpec(name='anesthesia', doc='anesthesia', dtype='text', required=False),
        NWBAttributeSpec(name='analgesics', doc='analgesics', dtype='text', required=False),
        NWBAttributeSpec(name='antibiotics', doc='antibiotics', dtype='text', required=False),
        NWBAttributeSpec(name='target_anatomy', doc='target anatomy', dtype='text', required=False),
        NWBAttributeSpec(name='room', doc='place where the surgery took place', dtype='text',
                         required=False),
        NWBAttributeSpec(name='surgery_type', doc='"chronic" or "acute"', dtype='text', required=False),
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Information about surgery')
    ])

surgeries = NWBGroupSpec(
    neurodata_type_def='Surgeries',
    neurodata_type_inc='NWBDataInterface',
    name='surgeries',
    doc='relevant data for surgeries', quantity='?',
    groups=[surgery],
    attributes=[
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Container for surgeries')
    ])

histology = NWBGroupSpec(
    neurodata_type_def='Histology',
    neurodata_type_inc='NWBDataInterface',
    name='histology',
    doc='information about histology of subject',
    quantity='?',
    attributes=[
        NWBAttributeSpec(name='file_name', doc='filename of histology images', dtype='text'),
        NWBAttributeSpec(name='file_name_ext', doc='filename extension', dtype='text'),
        NWBAttributeSpec(name='imaging_technique',
                         doc='histology imaging technique (e.g. widefield, confocal, etc.)',
                         dtype='text'),
        NWBAttributeSpec(name='slice_plane', doc='[Coronal, Sagital, Transverse, Other]',
                         required=False, dtype='text'),
        NWBAttributeSpec(name='slice_thickness', doc='thickness of slice (um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='location_along_axis', doc='Axis orthogal to SlicePlane (mm)',
                         dtype='float', required=False),
        NWBAttributeSpec(name='brain_region_target', doc='Allen Institute acronym',
                         dtype='text', required=False),
        NWBAttributeSpec(name='stainings', doc='stainings', dtype='text', required=False),
        NWBAttributeSpec(name='light_source', doc='wavelength of light source in nm',
                         dtype='float', required=False),
        NWBAttributeSpec(name='image_scale', doc='scale of image (pixels/100um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='scale_bar', doc='size of image scale bar (um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='post_processing', doc='[Z-stacked, Stiched]', dtype='text',
                         required=False),
        NWBAttributeSpec(name='user', doc='person involved', dtype='text', required=False),
        NWBAttributeSpec(name='notes', doc='anything else', dtype='text', required=False),
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Information about Histology')
    ])


subject = NWBGroupSpec(
    neurodata_type_inc='Subject',
    neurodata_type_def='BuzSubject',
    name='subject',
    doc='information about subject',
    groups=[surgeries, histology],
    attributes=[
        NWBAttributeSpec(
            name='sex', required=False, dtype='text',
            doc='Sex of subject. Options: "M": male, "F": female, "O": other, "U": unknown'),
        NWBAttributeSpec(name='species', doc='Species of subject', dtype='text',
                         required=False),
        NWBAttributeSpec(name='strain', dtype='text', doc='strain of animal',
                         required=False),
        NWBAttributeSpec(name='genotype', dtype='text', doc='genetic line of animal',
                         required=False),
        NWBAttributeSpec(name='date_of_birth', dtype='text', doc='in ISO 8601 format',
                         required=False),
        NWBAttributeSpec(name='date_of_death', dtype='text', doc='in ISO 8601 format',
                         required=False),
        NWBAttributeSpec(name='age', doc='age of subject. No specific format enforced.',
                         dtype='text', required=False),
        NWBAttributeSpec(name='gender', dtype='text', required=False,
                         doc='Gender of subject if different from sex.'),
        NWBAttributeSpec(name='earmark', dtype='text', required=False,
                         doc='Earmark of subject'),
        NWBAttributeSpec(name='weight', required=False, dtype='text',
                         doc='Weight at time of experiment, at time of surgery and at other '
                             'important times'),
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Buzsaki subject structure')
    ])

optical_fiber = NWBGroupSpec(
    neurodata_type_inc='Device',
    neurodata_type_def='OpticalFiber',
    name='OpticalFiber',
    doc='Meta-data about optical fiber',
    attributes=[
        NWBAttributeSpec(name='type', doc='model', dtype='text', required=False),
        NWBAttributeSpec(name='core_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='outer_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive', doc='whether a microdrive was used (0: not used, 1: used)',
                         dtype='int'),
        NWBAttributeSpec(name='microdrive_lead', doc='um/turn', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive_id', doc='id of microdrive', dtype='int', required=False),
        NWBAttributeSpec(name='help', doc='help', dtype='text', value='Information about optical fiber')
    ]
)

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)

specs = (subject, optical_fiber)
for spec in specs:
    ns_builder.add_spec(ext_source, spec)
ns_builder.export(ns_path)


def obj2docval(spec):

    args_spec = []

    for attrib in spec.attributes:
        if attrib.dtype is 'text':
            _type = str
        else:
            _type = attrib.dtype

        arg_spec = {'name': attrib.name, 'type': _type, 'doc': attrib.doc}
        if not attrib.required:
            arg_spec['default'] = None
        if not attrib.name == 'help':
            args_spec.append(arg_spec)

    for group in spec.groups + spec.datasets:
        arg_spec = {'name': group.name, 'type': group.neurodata_type_def, 'doc': group.doc}
        if group.quantity in ('?', '*'):
            arg_spec['default'] = None

        args_spec.append(arg_spec)

    names = [x['name'] for x in args_spec]
    super_args = eval(spec['neurodata_type_inc']).__init__.__docval__['args']
    for x in super_args:
        if x['name'] not in names:
            args_spec.append(x)

    return tuple(args_spec)


def get_nwbfields(spec):
    vars = [attrib.name for attrib in spec.attributes] + \
           [attrib.name for attrib in spec.datasets] + \
           [{'name': attrib.name, 'child': True} for attrib in spec.groups]

    return tuple(vars)

####


load_namespaces(ns_path)

# load custom classes
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)


def get_class(spec):
    class AutoClass(eval(spec['neurodata_type_inc'])):
        __nwbfields__ = get_nwbfields(spec)

        @docval(*obj2docval(spec))
        def __init__(self, **kwargs):
            super_args = [x['name'] for x in super(AutoClass, self).__init__.__docval__['args']]
            super(AutoClass, self).__init__(**{arg: kwargs[arg] for arg in super_args if arg in kwargs})
            for attr, val in kwargs.items():
                if attr not in super_args:
                    setattr(self, attr, val)
    return AutoClass


def camel2underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_multi_container(spec):
    inner_class_name =spec.groups[0]['neurodata_type_def']
    inner_class = camel2underscore(inner_class_name)
    InnerClass = eval(inner_class_name)

    @register_class(spec['neurodata_type_def'], name)
    class AutoClass(MultiContainerInterface):
        __clsconf__ = {
            'attr': inner_class + 's',
            'type': InnerClass,
            'add': 'add_' + inner_class,
            'get': 'get_' + inner_class,
            'create': 'create_' + inner_class,
        }

        __help = 'container for ' + inner_class + 's'

    return AutoClass


@register_class('Surgery', name)
class Surgery(get_class(surgery)):
    pass


@register_class('Surgeries', name)
class Surgeries(get_multi_container(surgeries)):
    pass


@register_class('VirusInjection', name)
class VirusInjection(get_class(virus_injection)):
    pass


@register_class('VirusInjections', name)
class VirusInjections(get_multi_container(virus_injections)):
    pass


@register_class('BuzSubject', name)
class BuzSubject(get_class(subject)):
    pass


@register_class('Histology', name)
class Histology(get_class(histology)):
    pass


@register_class('OpticalFiber', name)
class OpticalFiber(get_class(optical_fiber)):
    pass


virus_injections = VirusInjections(source='lab notebook')


virus_injections.add_virus_injection(
    VirusInjection(name='virus_injection1', coordinates=[1., 2., 3.], virus='a', volume=.45,
                   source='source'))

implantation = Surgery(name='implantation', notes='test surgery', source='lab notebook',
                       virus_injections=virus_injections)
surgeries = Surgeries(source='lab notebook')
surgeries.add_surgery(implantation)

subject = BuzSubject(subject_id='007', genotype='mouse1', species='mouse',
                     sex='U', age='3 months', surgeries=surgeries, source='notebook')

nwbfile = NWBFile("source", "a file with metadata", "NB123A", '2018-06-01T00:00:00', subject=subject)

nwbfile.add_device(OpticalFiber(microdrive=0, source='source', name='optical_fiber1'))

fname = 'test_ext.nwb'
with NWBHDF5IO(fname, 'w') as io:
    io.write(nwbfile)

with NWBHDF5IO(fname, 'r') as io:
    io.read()

