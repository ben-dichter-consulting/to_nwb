from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec

name = 'subject2'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

surgery = NWBGroupSpec(name='Surgery',
                       doc='relevant data for surgery',
                       datasets=[
                           NWBDatasetSpec(name='date', doc='date in ISO 8601', dtype='str'),
                           NWBDatasetSpec(name='notes', doc='notes', dtype='str'),
                           NWBDatasetSpec(name='pharmacology', doc='pharmacology',
                                          dtype='str'),
                           NWBDatasetSpec(name='target_anatomy', doc='target anatomy',
                                          dtype='str')])

implantation = NWBGroupSpec(doc='implantation', name='implantation',
                            neurodata_type_inc='Surgery',
                            datasets=[
                                NWBDatasetSpec(name='devices', doc='links to devices', dtype='RefSpec')
                            ])
virus_injection = NWBGroupSpec(name='VirusInjection',
                               doc='notes about surgery that includes virus injection',
                               datasets=[
                                   NWBDatasetSpec(name='virus_id', doc='id of virus', dtype='str'),
                                   NWBDatasetSpec(name='coordinates',
                                                  doc='(AP, ML, DV) of virus injection',
                                                  dtype='float',
                                                  shape=(3,)),
                                   NWBDatasetSpec(name='volume',
                                                  doc='volume of injecting in microliters',
                                                  dtype='float')
                               ])

subject = NWBGroupSpec(neurodata_type_def='Subject',
                       doc='information about subject',
                       groups=[surgery],
                       datasets=[
                           NWBDatasetSpec(name='genotype', dtype='str', doc='genotype'),
                           NWBDatasetSpec(name='date_of_birth', dtype='str',
                                          doc='ISO 8601', quantity='?'),
                           NWBDatasetSpec(name='age', doc='age of subject. No specific format enforced.',
                                          dtype='str', quantity='?'),
                           NWBDatasetSpec(name='sex', doc='Sex of subject. '
                                                          'Options: "M": male, "F": female, "O": other, "U": unknown',
                                          quantity='?',
                                          dtype='str'),
                           NWBDatasetSpec(name='gender', doc='gender of subject, if different from sex. Same options as'
                                                             ' sex.',
                                          dtype='str', quantity='?'),
                           NWBDatasetSpec(name='species', doc='Species of subject', dtype='str',
                                          quantity='?'),
                           NWBDatasetSpec(name='weight',
                                          doc='Weight at time of experiment, at time of surgery and at other important '
                                              'times',
                                          quantity='?',
                                          dtype='str'),
                           NWBDatasetSpec(name='subject_id', doc='ID of animal/person used/participating in experiment '
                                                                 '(lab convention)',
                                          quantity='?',
                                          dtype='str')
                       ])

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
for dtype in [surgery, implantation, virus_injection, subject]:
    ns_builder.add_spec(ext_source, dtype)
ns_builder.export(ns_path)
