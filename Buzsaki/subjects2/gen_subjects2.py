from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec

import pendulum

name = 'subject2'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

surgery = NWBGroupSpec(
    neurodata_type_def='Surgery',
    name='surgery', doc='relevant data for surgery', quantity='?',
    datasets=[NWBDatasetSpec(name='devices', quantity='?',
                             doc='links to implanted/explanted devices',
                             dtype=RefSpec('Device', 'object'))],
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
        NWBAttributeSpec(name='type', doc='"chronic" or "acute"', dtype='text', required=False)
    ])

virus_injection = NWBGroupSpec(
    neurodata_type_inc='Surgery',
    neurodata_type_def='VirusInjection',
    name='VirusInjection', quantity='?',
    doc='notes about surgery that includes virus injection',
    datasets=[NWBDatasetSpec(name='coordinates', doc='(AP, ML, DV) of virus injection',
                             dtype='float', shape=(3,))],
    attributes=[
        NWBAttributeSpec(name='virus', doc='type of virus', dtype='text'),
        NWBAttributeSpec(name='volume', doc='volume of injecting in nL', dtype='float'),
        NWBAttributeSpec(name='rate', doc='rate of injection (nL/s)',
                         dtype='float', required=False),
        NWBAttributeSpec(name='scheme', doc='scheme of injection', dtype='text', required=False)])

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
        NWBAttributeSpec(name='slice_thickness', docs='thickness of slice (um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='location_along_axis', docs='Axis orthogal to SlicePlane (mm)',
                         dtype='float', required=False),
        NWBAttributeSpec(name='brain_region_target', docs='Allen Institute acronym',
                         dtype='text', required=False),
        NWBAttributeSpec(name='stainings', docs='stainings', dtype='text', required=False),
        NWBAttributeSpec(name='light_source', docs='wavelength of light source in nm',
                         dtype='float', required=False),
        NWBAttributeSpec(name='image_scale', docs='scale of image (pixels/100um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='scale_bar', docs='size of image scale bar (um)', dtype='float',
                         required=False),
        NWBAttributeSpec(name='post_processing', docs='[Z-stacked, Stiched]', dtype='text',
                         required=False),
        NWBAttributeSpec(name='user', docs='person involved', dtype='text', required=False),
        NWBAttributeSpec(name='notes', docs='anything else', dtype='text', required=False)
    ])


subject = NWBGroupSpec(
    neurodata_type_def='Subject2',
    name='Subject2',
    doc='information about subject',
    groups=[surgery, histology],
    attributes=[
        NWBAttributeSpec(name='subject_id', required=True, dtype='text',
                         doc='ID of subject (lab convention)'),
        NWBAttributeSpec(
            name='sex', quantity='?', dtype='text', required=True,
            doc='Sex of subject. Options: "M": male, "F": female, "O": other, "U": unknown'),
        NWBAttributeSpec(name='species', doc='Species of subject', dtype='text',
                         quantity='?'),
        NWBAttributeSpec(name='strain', dtype='text', doc='strain of animal',
                         required=False),
        NWBAttributeSpec(name='genetic_line', dtype='text', doc='genetic line of animal',
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
    attributes=[
        NWBAttributeSpec(name='type', doc='model', dtype='text', required=False),
        NWBAttributeSpec(name='core_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='outer_diameter', doc='in um', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive', doc='whether a microdrive was used',
                         dtype='logical'),
        NWBAttributeSpec(name='microdrive_lead', doc='um/turn', dtype='float', required=False),
        NWBAttributeSpec(name='microdrive_id', doc=' ', dtype='int', required=False)
    ]
)

# for manipulation
#NWBAttributeSpec(name='rotation', doc='rotation of microdrive (degrees)',
#                 dtype='float', required=False),


ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
ns_builder.add_spec(ext_source, subject)
ns_builder.export(ns_path)

#######

from pynwb import load_namespaces, get_class

# load custom classes
name = 'subject2'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)

Subject = get_class('Subject2', name)
Surgery = get_class('Surgery', name)
VirusInjection = get_class('VirusInjection', name)

date = pendulum.now().to_iso8601_string()

Subject(genotype='mouse1', species='mouse', sex='U', subject_id='007',
        surgery=Surgery(
            date=date,
            virus_injection=VirusInjection(
                coordinates=[0.3, 0.2, 0.1],
                virus_id='123',
                volume=0.3
            )
        ))


