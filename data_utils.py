import numpy as np
import librosa

def normalize(x, dBFS_level=-26):
    scaling_factor = 10**(dBFS_level/20)
    return scaling_factor*x/(np.max(np.abs(x)))
    
def remove_silence(x, top_db=30):
    clip = librosa.effects.trim(x, top_db=top_db)
    return clip[0]