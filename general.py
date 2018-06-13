from collections import Iterable

from functools import partialmethod

from pynwb import load_namespaces, register_class
from pynwb.core import NWBDataInterface

from pynwb.form.utils import docval, popargs


from pynwb.form.backends.hdf5 import H5DataIO as gzip
gzip.__init__ = partialmethod(gzip.__init__, compress=True)

# load custom classes
name = 'general'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)


@register_class('CatCellInfo', name)
class CatCellInfo(NWBDataInterface):
    __nwbfields__ = ('values', 'indices', 'cell_index')

    @docval({'name': 'name', 'type': str, 'doc': 'name'},
            {'name': 'source', 'type': str, 'doc': 'source?'},
            {'name': 'values', 'type': Iterable, 'doc': 'unique values as strings'},
            {'name': 'indices', 'type': Iterable, 'doc': 'indexes into those values'},
            {'name': 'cell_index', 'type': Iterable,  'default': None,
             'doc': 'global id for neuron'})
    def __init__(self, **kwargs):
        name, source, values, indices, cell_index = popargs(
            'name', 'source', 'values', 'indices', 'cell_index', kwargs)
        super(CatCellInfo, self).__init__(name, source, **kwargs)

        self.values = values
        self.indices = indices
        if cell_index is not None:
            self.cell_index = cell_index




