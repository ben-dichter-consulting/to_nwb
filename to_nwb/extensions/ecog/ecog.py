import os

from pynwb import load_namespaces
from ..auto_class import get_class, get_multi_container

filepath = os.path.realpath(__file__)
basedir = os.path.split(filepath)[0]
name = 'ecog'

load_namespaces(os.path.join(basedir, name + '.namespace.yaml'))


Surface = get_class(name, 'Surface')

CorticalSurfaces = get_multi_container(name, 'CorticalSurfaces', Surface)
