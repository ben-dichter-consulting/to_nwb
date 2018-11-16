from scipy.io import loadmat
import numpy as np
from h5py import File


def load_wavs(raw_path):
    try:
        return load_wavs_mat(raw_path)
    except:
        return load_wavs_h5py(raw_path)


def load_wavs_mat(raw_path):
    out = []
    raw_matin = loadmat(raw_path, struct_as_record=True)
    streams = raw_matin['data']['streams'][0][0]
    wav_stream_names = sorted([x for x in streams.dtype.names if x[:3] == 'Wav'])
    for stream in wav_stream_names:
        out.append(streams[stream][0, 0]['data'][0, 0])
    fs = streams[stream][0, 0]['fs'][0, 0][0, 0]
    data = np.vstack(out).T
    return fs, data


def load_wavs_h5py(raw_path):

    out = []
    with File(raw_path) as f:
        streams = f['data']['streams']
        wav_stream_names = sorted([x for x in streams.keys() if x[:3] == 'Wav'])
        for stream in wav_stream_names:
            out.append(streams[stream]['data'][:])
        fs = streams[stream]['fs'][:][0, 0]
    data = np.hstack(out)
    return fs, data


def load_anin(raw_path, num=None):
    raw_matin = loadmat(raw_path, struct_as_record=True)
    anin_stream = raw_matin['data']['streams'][0, 0]['ANIN'][0, 0]
    data = anin_stream['data'][0, 0].T
    if num:
        data = data[:, num - 1]
    fs = anin_stream['fs'][0, 0][0, 0]
    return fs, data
