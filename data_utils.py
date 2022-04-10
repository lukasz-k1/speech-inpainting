import numpy as np
import librosa
import requests
from zipfile import ZipFile
import glob
import signal_corruption
import numpy as np
import pathlib
import soundfile
import os

def get_data(folder_path, source="url"):

    if source == "url":
        url = "https://datashare.ed.ac.uk/download/DS_10283_2651.zip"

        data = requests.get(url)

        with open(f"{folder_path}VCTK-Corpus.zip", 'wb') as file:
            file.write(data.content)

    with ZipFile(f'{folder_path}VCTK-Corpus.zip', 'r') as zipObj:
        zipObj.extractall()

    with ZipFile(f'{folder_path}VCTK-Corpus/VCTK-Corpus.zip', 'r') as zipObj:
        zipObj.extractall()



def normalize(x, dBFS_level=-26):
    scaling_factor = 10**(dBFS_level/20)
    return scaling_factor*x/(np.max(np.abs(x)))
    
def remove_silence(x, top_db=30):
    clip = librosa.effects.trim(x, top_db=top_db)
    return clip[0]


def cmvn(mfcc):
    stdevs = np.std(mfcc,1)
    means = np.mean(mfcc,1)
    return ((mfcc.T - means)/stdevs).transpose(), stdevs, means


def preprocessing(DATASET_PATH):

    data_dir = pathlib.Path(DATASET_PATH)

    speakers = np.array(os.listdir(str(data_dir)))

    filenames = glob.glob(str(data_dir) + '/*')

    paths_dict = dict(zip(speakers,[glob.glob(str(speaker_path) + '/*') for speaker_path in filenames]))

    for label, speaker_paths in paths_dict.items():
        print(f"Folder {label} processed") 
        for path in speaker_paths:

            if not path.endswith('.wav'):
                continue
            
            data, sr = librosa.load(path, sr=None)
            data = normalize(data)
            data = librosa.resample(data, sr, 16000)
            data = remove_silence(data)

            soundfile.write(path, data, 16000)


def extract_features(filepath):

    signal, fs = librosa.load(filepath, sr=None)
    
    signal = librosa.effects.preemphasis(signal, coef=0.97)

    corrupted_signal = signal_corruption.corrupt_signal(signal)

    signal_mfcc = librosa.feature.mfcc(signal, fs, n_mfcc=32, lifter=0.6, n_fft=640, hop_length=320, fmin=20, fmax=8000, n_mels=128)
    corrupted_signal_mfcc = librosa.feature.mfcc(corrupted_signal, fs, n_mfcc=32, lifter=0.6, n_fft=640, hop_length=320, fmin=20, fmax=8000, n_mels=128)

    signal_mfcc, signal_stds, signal_means = cmvn(signal_mfcc)
    corrupted_signal_mfcc, corrupted_signal_stds, corrupted_signal_means = cmvn(corrupted_signal_mfcc)

    return signal_mfcc, signal_stds, signal_means, corrupted_signal_mfcc, corrupted_signal_stds, corrupted_signal_means


