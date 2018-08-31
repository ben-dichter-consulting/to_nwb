import os

from pynwb import load_namespaces
from ..auto_class import get_class

filepath = os.path.realpath(__file__)
basedir = os.path.split(filepath)[0]
name = 'ecog'

load_namespaces(os.path.join(basedir, name + '.namespace.yaml'))


CorticalSurface = get_class(name, 'CorticalSurface')
