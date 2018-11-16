from scipy.io import loadmat
import numpy as np


def load_wavs(raw_path):
    out = []
    raw_matin = loadmat(raw_path, struct_as_record=True)
    wav_streams = raw_matin['data']['streams'][0][0]
    stream_names = wav_streams.dtype.names
    wav_stream_names = sorted([x for x in stream_names if x[:3] == 'Wav'])
    for stream in wav_stream_names:
        out.append(wav_streams[stream][0, 0]['data'][0, 0])
    fs = wav_streams[stream][0, 0]['fs'][0, 0][0, 0]
    return fs, np.vstack(out).T


def load_anin(raw_path, num=None):
    raw_matin = loadmat(raw_path, struct_as_record=True)
    anin_stream = raw_matin['data']['streams'][0, 0]['ANIN'][0, 0]
    data = anin_stream['data'][0, 0].T
    if num:
        data = data[:, num - 1]
    fs = anin_stream['fs'][0, 0][0, 0]
    return fs, data
