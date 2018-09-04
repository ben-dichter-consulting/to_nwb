from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec

name = 'ecog'
ns_path = name + ".namespace.yaml"
ext_source = name + ".extensions.yaml"

surface = NWBGroupSpec(neurodata_type_def='Surface',
                       neurodata_type_inc='NWBDataInterface',
                       quantity='+',
                       doc='brain cortical surface',
                       datasets=[  # set Faces and Vertices as elements of the Surfaces neurodata_type
                           NWBDatasetSpec(doc='faces for surface, indexes vertices', shape=(None, 3),
                                          name='faces', dtype='uint', dims=('face_number', 'vertex_index')),
                           NWBDatasetSpec(doc='vertices for surface, points in 3D space', shape=(None, 3),
                                          name='vertices', dtype='float', dims=('vertex_number', 'xyz'))])

surfaces = NWBGroupSpec(neurodata_type_def='CorticalSurfaces',
                        neurodata_type_inc='NWBDataInterface',
                        name='cortical_surfaces',
                        doc='triverts for cortical surfaces', quantity='?',
                        groups=[surface])

ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
ns_builder.add_spec(ext_source, surfaces)
ns_builder.export(ns_path)
