from pynwb import load_namespaces, get_class

# load custom classes
name = 'subject2'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)

Subject = get_class('Subject2', name)

Subject(genotype='mouse1', species='mouse', )