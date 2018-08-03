from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec
from pynwb import register_class, load_namespaces, NWBFile, NWBHDF5IO, get_class
from pynwb.form.utils import docval
from pynwb.file import Subject as original_Subject, NWBContainer, MultiContainerInterface


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
        NWBAttributeSpec(name='scheme', doc='scheme of injection', dtype='text', required=False)])

surgery = NWBGroupSpec(
    neurodata_type_def='Surgery', doc='information about a specific surgery', quantity='+',
    neurodata_type_inc='NWBDataInterface',
    datasets=[NWBDatasetSpec(name='devices', quantity='?', doc='links to implanted/explanted devices',
                             dtype=RefSpec('Device', 'object'))],
    groups=[virus_injection],
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
        NWBAttributeSpec(name='surgery_type', doc='"chronic" or "acute"', dtype='text', required=False)])

surgeries = NWBGroupSpec(neurodata_type_def='Surgeries', name='surgeries',
                         doc='relevant data for surgeries', quantity='?',
                         groups=[surgery])

histology = NWBGroupSpec(
    neurodata_type_def='Histology',
    name='Histology',
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
        NWBAttributeSpec(name='notes', doc='anything else', dtype='text', required=False)
    ])


subject = NWBGroupSpec(
    neurodata_type_inc='Subject',
    neurodata_type_def='buz_Subject',
    name='subject',
    doc='information about subject',
    groups=[surgeries, histology],
    attributes=[
        NWBAttributeSpec(name='subject_id', required=True, dtype='text',
                         doc='ID of subject (lab convention)'),
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
                             'important times')])

OpticalFiber = NWBGroupSpec(
    neurodata_type_inc='Device',
    neurodata_type_def='OpticalFiber',
    name='OpticalFiber',
    doc='Meta-data about optical fiber',
    attributes=[
        NWBAttributeSpec(name='type', doc='model', dtype='text', required=False),
        NWBAttributeSpec(name='core_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='outer_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive', doc='whether a microdrive was used (0: not used, 1: used)',
                         dtype='uint'),
        NWBAttributeSpec(name='microdrive_lead', doc='um/turn', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive_id', doc='id of microdrive', dtype='int', required=False)
    ]
)

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)

specs = (subject, OpticalFiber)
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

        args_spec.append(arg_spec)

    for group in spec.groups:
        arg_spec = {'name': group.name, 'type': group.neurodata_type_def, 'doc': group.doc}
        if group.quantity == '?':
            arg_spec['default'] = None

        args_spec.append(arg_spec)

    return tuple(args_spec)


def get_nwbfields(spec):
    vars = [attrib.name for attrib in spec.attributes] + \
           [attrib.name for attrib in spec.groups] + \
           [attrib.name for attrib in spec.datasets]

    return tuple(vars)

####


load_namespaces(ns_path)


@register_class('buz_Subject', name)
class mySubject(original_Subject):

    __nwbfields__ = tuple(list(get_nwbfields(subject)) + ['surgeries'])

    @docval(*obj2docval(subject))
    def __init__(self, **kwargs):

        original_Subject.__init__(self, subject_id=kwargs['subject_id'], source='source')

        for attr, val in kwargs.items():
            if attr is not 'subject_id':
                setattr(self, attr, val)


# load custom classes
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)

Surgery = get_class('Surgery', name)


@register_class('Surgeries', name)
class Surgeries(MultiContainerInterface):

    __clsconf__ = {
        'attr': 'surgerys',
        'type': Surgery,
        'add': 'add_surgery',
        'get': 'get_surgery',
        'create': 'create_surgery',
    }

    __help = 'info about surgeries'


surgeries = Surgeries(source='lab notebook')
surgery = Surgery(name='implantation', notes='test surgery', source='lab notebook')
surgeries.add_surgery(surgery)

subject = mySubject(subject_id='007', genotype='mouse1', species='mouse',
                    sex='U', age='3 months', surgeries=surgeries)

nwbfile = NWBFile("source", "a file with metadata", "NB123A", '2018-06-01T00:00:00', subject=subject)

fname = 'test_ext.nwb'
with NWBHDF5IO(fname, 'w') as io:
    io.write(nwbfile)

