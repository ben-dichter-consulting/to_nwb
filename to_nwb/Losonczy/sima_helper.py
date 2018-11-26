import os
import pickle
import sima


def get_motion_correction(expt):
    fpath = os.path.join(expt.sima_path(), 'sequences.pkl')

    with open(fpath, 'rb') as f:
        aa = pickle.load(f)

    obj = aa[0]

    while True:
        if 'displacements' in obj:
            return obj['displacements']
        obj = obj['base']
