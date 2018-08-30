from pynwb.form.utils import docval
from pynwb import __NS_CATALOG, register_class

from pynwb.ecephys import *
from pynwb.icephys import *
from pynwb.file import *
from pynwb.device import *
from pynwb.misc import *
from pynwb.ophys import *
from pynwb.ogen import *
from pynwb.retinotopy import *
from pynwb.behavior import *


def obj2docval(spec):

    args_spec = []

    for attrib in spec.attributes:
        if 'shape' in attrib:
            _type = list
        elif attrib.dtype == 'text':
            _type = str
        else:
            _type = attrib.dtype

        arg_spec = {'name': attrib.name, 'type': _type, 'doc': attrib.doc}
        if 'value' in attrib:
            arg_spec['default'] = attrib.value
        elif not attrib.required:
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
    vars = [attrib.name for attrib in spec.attributes if attrib.name]
    if hasattr(spec, 'datasets'):
        vars += [dataset.name for dataset in spec.datasets if dataset.name]

    if hasattr(spec, 'groups'):
        for attrib in spec.groups:
            if attrib.name:
                if 'neurodata_type_inc' in attrib or 'neurodata_type_def' in attrib:
                    vars.append({'name': attrib.name, 'child': True})
                else:
                    vars.append(attrib.name)

    return tuple(vars)


def get_class(namespace, data_type):
    spec = __NS_CATALOG.get_spec(namespace, data_type)
    __nwbfields__ = get_nwbfields(spec)

    @docval(*obj2docval(spec))
    def __init__(self, **kwargs):
        super_args = [x['name'] for x in super(type(self), self).__init__.__docval__['args']]
        super(type(self), self).__init__(**{arg: kwargs[arg] for arg in super_args
                                            if arg in kwargs and kwargs[arg] is not None})
        [setattr(self, attr, val) for attr, val in kwargs.items() if not hasattr(self, attr)]

    d = {'__init__': __init__, '__nwbfields__': __nwbfields__}

    cls = type(spec['neurodata_type_def'], (eval(spec['neurodata_type_inc']),), d)
    register_class(data_type, namespace,  cls)
    return cls
