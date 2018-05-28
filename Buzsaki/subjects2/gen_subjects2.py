from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec, NWBAttributeSpec
from pynwb.form.spec import RefSpec

name = 'subject2'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

virus_injection = NWBGroupSpec(neurodata_type_def='VirusInjection',
                               name='VirusInjection',
                               doc='notes about surgery that includes virus injection',
                               datasets=[
                                   NWBDatasetSpec(name='coordinates',
                                                  doc='(AP, ML, DV) of virus injection',
                                                  dtype='float',
                                                  shape=(3,)),
                                   NWBDatasetSpec(name='volume',
                                                  doc='volume of injecting in microliters',
                                                  dtype='float')
                               ],
                               attributes=[NWBAttributeSpec(name='virus_id', doc='id of virus', dtype='text')])

surgery = NWBGroupSpec(neurodata_type_def='Surgery',
                       name='Surgery',
                       doc='relevant data for surgery',
                       quantity='?',
                       datasets=[
                           NWBDatasetSpec(name='devices', doc='links to devices', dtype=RefSpec('Device', 'object'),
                                          quantity='?')
                       ],
                       groups=[virus_injection],
                       attributes=[NWBAttributeSpec(name='date', doc='date in ISO 8601', dtype='text'),
                                   NWBAttributeSpec(name='notes', doc='notes', dtype='str'),
                                   NWBAttributeSpec(name='pharmacology', doc='pharmacology', dtype='text'),
                                   NWBAttributeSpec(name='target_anatomy', doc='target anatomy', dtype='text')])


subject = NWBGroupSpec(neurodata_type_def='Subject2',
                       name='Subject2',
                       doc='information about subject',
                       groups=[surgery],
                       attributes=[NWBAttributeSpec(name='genotype', dtype='text', doc='genotype', quantity='?'),
                                   NWBAttributeSpec(name='date_of_birth', dtype='text', doc='ISO 8601', quantity='?'),
                                   NWBAttributeSpec(name='age', doc='age of subject. No specific format enforced.',
                                                    dtype='str', quantity='?'),
                                   NWBAttributeSpec(name='sex', quantity='?', dtype='text',
                                                    doc='Sex of subject. Options: "M": male, "F": female, "O": other, '
                                                        '"U": unknown'),
                                   NWBAttributeSpec(name='gender', dtype='text', quantity='?',
                                                    doc='Gender of subject if different from sex.'),
                                   NWBAttributeSpec(name='species', doc='Species of subject', dtype='text',
                                                    quantity='?'),
                                   NWBAttributeSpec(name='weight', quantity='?', dtype='text',
                                                    doc='Weight at time of experiment, at time of surgery and at other '
                                                        'important times'),
                                   NWBAttributeSpec(name='subject_id', quantity='?', dtype='text',
                                                    doc='ID of animal/person used/participating in experiment (lab '
                                                        'convention)'),
                                   NWBAttributeSpec(name='weight', quantity='?', dtype='text',
                                                    doc='Weight at time of experiment, at time of surgery and at other '
                                                        'important times')])


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

Subject(genotype='mouse1', species='mouse', )